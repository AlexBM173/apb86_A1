from __future__ import annotations

"""Data loading and I/O utilities for the coursework package.

This module provides functions to load observations and simulation data from disk,
and to persist normalised/transformed datasets and trained models in standard formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class ObservationData:
    """Container for observed 21-cm power-spectrum measurements.

    Attributes:
        k: One-dimensional array of wavenumber bins.
        power: One-dimensional array of observed power values aligned with `k`.
    """

    k: np.ndarray
    power: np.ndarray


@dataclass(frozen=True)
class SimulationData:
    """Container for the full simulation dataset and associated metadata.

    Attributes:
        ks: Per-simulation wavenumber grid.
        spectra: Per-simulation power spectra.
        params: Flattened model parameters per simulation (shape: n_samples x 4).
        redshift: Scalar redshift value per simulation.
        code: Simulation code identifier per sample.
        code_version: Simulation code version per sample.
        filenames: Source filename for each loaded simulation sample.
    """

    ks: np.ndarray
    spectra: np.ndarray
    params: np.ndarray
    redshift: np.ndarray
    code: list[str]
    code_version: list[str]
    filenames: list[str]


def _as_scalar(value: Any) -> float:
    """Convert scalar-like inputs to a Python float.

    Args:
        value: Value that should represent a scalar.

    Returns:
        Scalar value as `float`.

    Raises:
        ValueError: If the value cannot be interpreted as a scalar.
    """

    array = np.asarray(value)
    # Many fields are stored as zero-dimensional numpy arrays; treat them as
    # scalar values for downstream numeric processing.
    if array.shape == ():
        return float(array.item())
    if array.size == 1:
        return float(array.reshape(-1)[0])
    raise ValueError(f"Expected scalar value, got shape {array.shape}")


def _extract_param_dict(raw_value: Any, filename: str, expected_size: int) -> list[float]:
    """Extract and validate scalar parameter values from serialized dictionaries.

    Args:
        raw_value: Object array entry that should contain a dictionary of params.
        filename: Source filename used for informative error messages.
        expected_size: Number of expected parameters in the dictionary.

    Returns:
        Parameter values converted to `float`.

    Raises:
        TypeError: If the serialized object is not a dictionary.
        ValueError: If dictionary cardinality does not match `expected_size`.
    """

    param_dict = np.asarray(raw_value, dtype=object).item()
    if not isinstance(param_dict, dict):
        raise TypeError(f"Expected parameter dictionary in {filename}, got {type(param_dict)!r}")

    values = [_as_scalar(value) for value in param_dict.values()]
    if len(values) != expected_size:
        raise ValueError(
            f"Expected {expected_size} parameters in {filename}, got {len(values)}"
        )
    return values


def load_observations(data_dir: str | Path) -> ObservationData:
    """Load observed spectrum from `observations.npz`.

    Args:
        data_dir: Directory containing `observations.npz`.

    Returns:
        ObservationData with wavenumbers and power values.
    """

    observations_path = Path(data_dir) / "observations.npz"
    with np.load(observations_path) as observations_data:
        return ObservationData(
            k=np.asarray(observations_data["k"], dtype=float),
            power=np.asarray(observations_data["power"], dtype=float),
        )


def load_simulation_dataset(data_dir: str | Path) -> SimulationData:
    """Load and materialise all simulation samples from disk.

    Reads every `.npz` file under `data/simulations/simulations`, extracts
    astrophysical and cosmological parameters from nested dictionaries,
    and aggregates all samples and metadata.

    Args:
        data_dir: Root data directory containing the simulations folder.

    Returns:
        SimulationData with arrays of spectra, parameters, and metadata.
    """

    simulations_dir = Path(data_dir) / "simulations" / "simulations"
    ks: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    params: list[list[float]] = []
    redshift: list[float] = []
    code: list[str] = []
    code_version: list[str] = []
    filenames: list[str] = []

    for sample_path in sorted(simulations_dir.glob("*.npz")):
        with np.load(sample_path, allow_pickle=True) as sample:
            # Keep filenames to preserve provenance for debugging/inspection.
            filenames.append(sample_path.name)
            ks.append(np.asarray(sample[sample.files[0]], dtype=float))
            spectra.append(np.asarray(sample[sample.files[1]], dtype=float))

            # The source data stores astrophysical and cosmological parameters as
            # serialized dictionaries inside object arrays.
            astrophysical_params = _extract_param_dict(
                sample[sample.files[2]], sample_path.name, expected_size=3
            )
            cosmological_params = _extract_param_dict(
                sample[sample.files[3]], sample_path.name, expected_size=1
            )
            params.append(astrophysical_params + cosmological_params)

            redshift.append(_as_scalar(sample[sample.files[4]]))
            code.append(str(np.asarray(sample[sample.files[5]]).item()))
            code_version.append(str(np.asarray(sample[sample.files[6]]).item()))

    return SimulationData(
        ks=np.asarray(ks, dtype=float),
        spectra=np.asarray(spectra, dtype=float),
        params=np.asarray(params, dtype=float),
        redshift=np.asarray(redshift, dtype=float),
        code=code,
        code_version=code_version,
        filenames=filenames,
    )


def save_split_datasets(
    output_dir: str | Path,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Persist train/validation/test splits as separate `.npz` files.

    Args:
        output_dir: Directory where split files will be written.
        x_train: Training feature matrix.
        y_train: Training target matrix.
        x_val: Validation feature matrix.
        y_val: Validation target matrix.
        x_test: Test feature matrix.
        y_test: Test target matrix.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(output_path / "train.npz", X_train=x_train, y_train=y_train)
    np.savez(output_path / "val.npz", X_val=x_val, y_val=y_val)
    np.savez(output_path / "test.npz", X_test=x_test, y_test=y_test)


def load_split_dataset(file_path: str | Path) -> dict[str, np.ndarray]:
    """Load one split file created by :func:`save_split_datasets`.

    Args:
        file_path: Path to one of `train.npz`, `val.npz`, or `test.npz`.

    Returns:
        Dictionary mapping saved array names to numpy arrays.
    """

    with np.load(file_path, allow_pickle=True) as split_data:
        return {key: np.asarray(split_data[key]) for key in split_data.files}


def save_observations_pca(output_path: str | Path, observations_pca: np.ndarray) -> None:
    """Save PCA-transformed observation vector to disk.

    Args:
        output_path: Destination `.npz` path.
        observations_pca: PCA-space observation array.
    """

    np.savez(Path(output_path), observations_pca=np.asarray(observations_pca, dtype=float))


def save_pca_model(output_path: str | Path, pca_model: Any) -> None:
    """Serialize a fitted PCA model with joblib.

    Args:
        output_path: Destination pickle path.
        pca_model: Fitted PCA-like object.
    """

    joblib.dump(pca_model, Path(output_path))