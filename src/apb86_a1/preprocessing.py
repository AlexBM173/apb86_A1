from __future__ import annotations

"""Preprocessing utilities for scaling, PCA projection, and dataset splitting."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class NormalizationStats:
    """Min/max statistics used for min-max normalization.

    Attributes:
        min_power: Global minimum across simulation spectra.
        max_power: Global maximum across simulation spectra.
    """

    min_power: float
    max_power: float

    @property
    def scale(self) -> float:
        """Return the min-max denominator used by normalization."""

        return self.max_power - self.min_power


@dataclass(frozen=True)
class PCAResults:
    """Outputs of fitting PCA and transforming simulations + observation.

    Attributes:
        pca_model: Fitted sklearn PCA object.
        simulation_components: PCA components for simulation spectra.
        observation_components: PCA components for the observed spectrum.
    """

    pca_model: PCA
    simulation_components: np.ndarray
    observation_components: np.ndarray


@dataclass(frozen=True)
class DatasetSplits:
    """Train/validation/test partition for features and targets."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    """Validate and coerce input arrays to 2D float arrays.

    Args:
        array: Candidate array to validate.

    Returns:
        A 2D `float` numpy array.

    Raises:
        ValueError: If `array` is not 2-dimensional.
    """

    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    return array


def normalize_spectra(
    spectra: np.ndarray,
    stats: NormalizationStats | None = None,
) -> tuple[np.ndarray, NormalizationStats]:
    """Apply min-max normalization to simulation spectra.

    Args:
        spectra: 2D simulation spectra array `(n_samples, n_features)`.
        stats: Optional precomputed normalization stats.

    Returns:
        Tuple of normalized spectra and the stats used.
    """

    spectra = _ensure_2d(spectra)
    if stats is None:
        # Use global extrema from simulations so the same scaling can be reused
        # for the observed spectrum.
        stats = NormalizationStats(
            min_power=float(np.min(spectra)),
            max_power=float(np.max(spectra)),
        )
    if stats.scale == 0:
        raise ValueError("Cannot normalize spectra when max_power equals min_power")
    normalized = (spectra - stats.min_power) / stats.scale
    return normalized, stats


def normalize_observation(observation_spectrum: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Normalize the observed spectrum with simulation-derived statistics.

    Args:
        observation_spectrum: One-dimensional observed power spectrum.
        stats: Min/max statistics from simulations.

    Returns:
        Normalized observation spectrum.
    """

    observation_spectrum = np.asarray(observation_spectrum, dtype=float)
    if observation_spectrum.ndim != 1:
        raise ValueError(
            f"Expected a 1D observation spectrum, got shape {observation_spectrum.shape}"
        )
    if stats.scale == 0:
        raise ValueError("Cannot normalize observation when max_power equals min_power")
    return (observation_spectrum - stats.min_power) / stats.scale


def fit_pca_with_observation(
    simulation_spectra: np.ndarray,
    observation_spectrum: np.ndarray,
    *,
    n_components: int = 2,
) -> PCAResults:
    """Fit PCA on simulations plus observation and transform both.

    The observation vector is appended during fitting so projection space is
    informed by both simulation and observed distributions.

    Args:
        simulation_spectra: Normalized 2D simulation spectra.
        observation_spectrum: Normalized 1D observed spectrum.
        n_components: Number of PCA dimensions to keep.

    Returns:
        PCA fit object with transformed simulations and observation.
    """

    simulation_spectra = _ensure_2d(simulation_spectra)
    observation_spectrum = np.asarray(observation_spectrum, dtype=float)
    if observation_spectrum.ndim != 1:
        raise ValueError(
            f"Expected a 1D observation spectrum, got shape {observation_spectrum.shape}"
        )

    # Append the observation as an extra row before fitting PCA.
    combined = np.vstack([simulation_spectra, observation_spectrum.reshape(1, -1)])
    pca_model = PCA(n_components=n_components).fit(combined)
    return PCAResults(
        pca_model=pca_model,
        simulation_components=pca_model.transform(simulation_spectra),
        observation_components=pca_model.transform(observation_spectrum.reshape(1, -1)),
    )


def cumulative_explained_variance(pca_model: PCA) -> np.ndarray:
    """Compute cumulative explained-variance ratio for a fitted PCA model."""

    return np.cumsum(pca_model.explained_variance_ratio_)


def split_training_data(
    params: np.ndarray,
    targets: np.ndarray,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_state: int = 42,
) -> DatasetSplits:
    """Split features/targets into train, validation, and test partitions.

    Args:
        params: Feature matrix.
        targets: Target matrix.
        train_fraction: Fraction for training data.
        val_fraction: Fraction for validation data.
        test_fraction: Fraction for test data.
        random_state: Random seed for deterministic splitting.

    Returns:
        Data container with six split arrays.
    """

    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction, val_fraction, and test_fraction must sum to 1")

    params = _ensure_2d(params)
    targets = _ensure_2d(targets)
    if len(params) != len(targets):
        raise ValueError("params and targets must have the same number of samples")

    # First split isolates the training portion.
    x_train, x_temp, y_train, y_temp = train_test_split(
        params,
        targets,
        test_size=(1.0 - train_fraction),
        random_state=random_state,
    )
    # Second split partitions the remainder into validation/test according to
    # their relative proportions.
    temp_test_fraction = test_fraction / (val_fraction + test_fraction)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=temp_test_fraction,
        random_state=random_state,
    )
    return DatasetSplits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )