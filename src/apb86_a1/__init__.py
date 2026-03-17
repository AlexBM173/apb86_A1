"""Public package API for APB86 A1 preprocessing and emulator workflows.

The package exposes:

- I/O utilities for loading simulation/observation data and saving artefacts.
- Preprocessing helpers for normalisation, PCA, and dataset splitting.
- Emulator utilities for model construction, training, evaluation, and
    hyperparameter optimisation.

British English spellings are used throughout (normalisation, optimisation, etc.).
American English aliases are provided for backwards compatibility.
"""

from .emulator import (
    EvaluationResult,
    NeuralNetworkEmulator,
    OptimisationResult,
    OptimizationResult,  # Backwards-compatibility alias
    TrainingConfig,
    TrainingHistory,
    build_emulator,
    optimise_emulator,
    optimize_emulator,  # Backwards-compatibility alias
    predict,
    test_emulator,
    train_emulator,
)
from .io import (
    ObservationData,
    SimulationData,
    load_observations,
    load_simulation_dataset,
    load_split_dataset,
    save_observations_pca,
    save_pca_model,
    save_split_datasets,
)
from .preprocessing import (
    DatasetSplits,
    NormalisationStats,
    NormalizationStats,  # Backwards-compatibility alias
    PCAResults,
    cumulative_explained_variance,
    fit_pca_with_observation,
    normalise_observation,
    normalize_observation,  # Backwards-compatibility alias
    normalise_spectra,
    normalize_spectra,  # Backwards-compatibility alias
    split_training_data,
)

__all__ = [
    "DatasetSplits",
    "EvaluationResult",
    "NeuralNetworkEmulator",
    "NormalisationStats",
    "NormalizationStats",  # Backwards compatibility
    "ObservationData",
    "OptimisationResult",
    "OptimizationResult",  # Backwards compatibility
    "PCAResults",
    "SimulationData",
    "TrainingConfig",
    "TrainingHistory",
    "build_emulator",
    "cumulative_explained_variance",
    "fit_pca_with_observation",
    "load_observations",
    "load_simulation_dataset",
    "load_split_dataset",
    "normalise_observation",
    "normalize_observation",  # Backwards compatibility
    "normalise_spectra",
    "normalize_spectra",  # Backwards compatibility
    "optimise_emulator",
    "optimize_emulator",  # Backwards compatibility
    "predict",
    "save_observations_pca",
    "save_pca_model",
    "save_split_datasets",
    "split_training_data",
    "test_emulator",
    "train_emulator",
]