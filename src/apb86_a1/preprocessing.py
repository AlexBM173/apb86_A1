from __future__ import annotations

"""Preprocessing utilities for scaling, PCA projection, and dataset splitting.

This module provides normalisation (min-max scaling), PCA fitting and transformation,
and dataset partitioning with proper separation to avoid data leakage. Normalisation
statistics are derived from the training set only and applied consistently to
validation, test, and observation data.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class NormalisationStats:
    """Min/max statistics used for min-max normalisation.

    Attributes:
        min_power: Minimum power value (typically from training set).
        max_power: Maximum power value (typically from training set).
    """

    min_power: float
    max_power: float

    @property
    def scale(self) -> float:
        """Return the min-max denominator used by normalisation."""

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


def normalise_spectra(
    spectra: np.ndarray,
    stats: NormalisationStats | None = None,
) -> tuple[np.ndarray, NormalisationStats]:
    """Apply min-max normalisation to simulation spectra.

    Spectra are scaled to [0, 1] using provided or computed min/max statistics.
    To avoid data leakage, statistics should be computed from the training set only
    and applied to validation, test, and observation spectra.

    Args:
        spectra: 2D simulation spectra array of shape `(n_samples, n_features)`.
        stats: Optional precomputed normalisation statistics (e.g. from training set).
               If None, statistics are computed from the input spectra.

    Returns:
        Tuple of (normalised spectra, normalisation statistics used).

    Raises:
        ValueError: If max_power equals min_power (cannot scale).
    """

    spectra = _ensure_2d(spectra)
    if stats is None:
        # Compute extrema from the provided spectra (typically training set).
        stats = NormalisationStats(
            min_power=float(np.min(spectra)),
            max_power=float(np.max(spectra)),
        )
    if stats.scale == 0:
        raise ValueError("Cannot normalise spectra when max_power equals min_power")
    normalised = (spectra - stats.min_power) / stats.scale
    return normalised, stats


def normalise_observation(observation_spectrum: np.ndarray, stats: NormalisationStats) -> np.ndarray:
    """Normalise the observed spectrum with training-derived statistics.

    The observation is scaled using min/max statistics computed from the training set,
    ensuring consistency with the training normalisation.

    Args:
        observation_spectrum: 1D observed power spectrum array.
        stats: Min/max statistics (typically from training set).

    Returns:
        Normalised observation spectrum.

    Raises:
        ValueError: If spectrum is not 1D or if max_power equals min_power.
    """

    observation_spectrum = np.asarray(observation_spectrum, dtype=float)
    if observation_spectrum.ndim != 1:
        raise ValueError(
            f"Expected a 1D observation spectrum, got shape {observation_spectrum.shape}"
        )
    if stats.scale == 0:
        raise ValueError("Cannot normalise observation when max_power equals min_power")
    return (observation_spectrum - stats.min_power) / stats.scale


def fit_pca_with_observation(
    simulation_spectra: np.ndarray,
    observation_spectrum: np.ndarray,
    *,
    n_components: int = 2,
) -> PCAResults:
    """Fit PCA on simulations plus observation and transform both.

    Fits PCA on normalised training spectra (including observation in fit data)
    to ground the projection space in both simulation and observed distributions.
    The model is then used to transform both simulations and observation.

    Args:
        simulation_spectra: Normalised 2D simulation spectra (typically training set).
        observation_spectrum: Normalised 1D observed spectrum.
        n_components: Number of PCA dimensions to keep.

    Returns:
        PCAResults with fitted PCA model and transformed components for both
        simulation and observation.

    Raises:
        ValueError: If spectrum dimensions do not match expected shapes.
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

    Performs a two-stage stratified split: first separates training data, then
    partitions the remainder into validation and test sets according to their
    relative proportions.

    Args:
        params: Feature matrix of shape `(n_samples, n_features)`.
        targets: Target matrix of shape `(n_samples, n_outputs)`.
        train_fraction: Fraction for training data (default: 0.8).
        val_fraction: Fraction for validation data (default: 0.1).
        test_fraction: Fraction for test data (default: 0.1).
        random_state: Random seed for deterministic splitting (default: 42).

    Returns:
        DatasetSplits container with six split arrays (x_train, y_train, x_val,
        y_val, x_test, y_test).

    Raises:
        ValueError: If fractions do not sum to 1.0 or array lengths are mismatched.
    """

    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction, val_fraction, and test_fraction must sum to 1")

    params = _ensure_2d(params)
    targets = _ensure_2d(targets)
    if len(params) != len(targets):
        raise ValueError("params and targets must have the same number of samples")

    # First split: isolate the training portion from the full dataset.
    x_train, x_temp, y_train, y_temp = train_test_split(
        params,
        targets,
        test_size=(1.0 - train_fraction),
        random_state=random_state,
    )
    # Second split: partition the remainder into validation and test sets
    # proportionally based on their relative sizes.
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


# Backwards-compatibility aliases for American English spellings.
NormalizationStats = NormalisationStats
normalize_spectra = normalise_spectra
normalize_observation = normalise_observation