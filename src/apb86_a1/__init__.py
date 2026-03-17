"""Public package API for APB86 A1 preprocessing and emulator workflows.

The package exposes:

- I/O utilities for loading simulation/observation data and saving artifacts.
- Preprocessing helpers for normalization, PCA, and dataset splitting.
- Emulator utilities for model construction, training, evaluation, and
    hyperparameter optimization.
"""

from .emulator import (
    EvaluationResult,
    NeuralNetworkEmulator,
    OptimizationResult,
    TrainingConfig,
    TrainingHistory,
    build_emulator,
    optimize_emulator,
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
    NormalizationStats,
    PCAResults,
    cumulative_explained_variance,
    fit_pca_with_observation,
    normalize_observation,
    normalize_spectra,
    split_training_data,
)

__all__ = [
    "DatasetSplits",
    "EvaluationResult",
    "NeuralNetworkEmulator",
    "NormalizationStats",
    "ObservationData",
    "OptimizationResult",
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
    "normalize_observation",
    "normalize_spectra",
    "optimize_emulator",
    "predict",
    "save_observations_pca",
    "save_pca_model",
    "save_split_datasets",
    "split_training_data",
    "test_emulator",
    "train_emulator",
]