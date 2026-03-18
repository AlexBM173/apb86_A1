# APB86 A1 Package Documentation

## Overview

The lba package provides an end-to-end workflow for:

- loading 21-cm observation and simulation data
- normalising spectra and applying PCA
- splitting datasets for machine-learning workflows
- training and evaluating a neural-network emulator
- running Optuna-based hyperparameter optimisation with architecture and optimiser search

The preprocessing flow is leakage-safe by design:

- split the simulation dataset into train, validation, and test first
- compute min/max power from the training spectra only
- apply the same normalisation statistics to validation, test, and observation spectra

## Installation and Execution

Use directly from the repository:

```bash
PYTHONPATH=src python3 -m lba --output-dir data
```

Or after installation:

```bash
lba --output-dir data
```

## Package Layout

- src/lba/io.py: data loading and persistence helpers
- src/lba/preprocessing.py: normalisation, PCA, and data splitting
- src/lba/emulator.py: model, training, evaluation, and optimisation
- src/lba/cli.py: end-to-end command-line pipeline
- src/lba/__init__.py: public API exports

## API Reference

### Module: lba.io

#### Data Classes

- ObservationData
  - k: np.ndarray
  - power: np.ndarray

- SimulationData
  - ks: np.ndarray
  - spectra: np.ndarray
  - params: np.ndarray
  - redshift: np.ndarray
  - code: list[str]
  - code_version: list[str]
  - filenames: list[str]

#### Public Functions

- load_observations(data_dir: str | Path) -> ObservationData
- load_simulation_dataset(data_dir: str | Path) -> SimulationData
- save_split_datasets(output_dir, *, x_train, y_train, x_val, y_val, x_test, y_test) -> None
- load_split_dataset(file_path: str | Path) -> dict[str, np.ndarray]
- save_observations_pca(output_path: str | Path, observations_pca: np.ndarray) -> None
- save_pca_model(output_path: str | Path, pca_model: Any) -> None

### Module: lba.preprocessing

#### Data Classes

- NormalisationStats
  - min_power: float
  - max_power: float
  - scale: float property

- PCAResults
  - pca_model: sklearn.decomposition.PCA
  - simulation_components: np.ndarray
  - observation_components: np.ndarray

- DatasetSplits
  - x_train, x_val, x_test, y_train, y_val, y_test

#### Public Functions

- normalise_spectra(spectra: np.ndarray, stats: NormalisationStats | None = None) -> tuple[np.ndarray, NormalisationStats]
- normalise_observation(observation_spectrum: np.ndarray, stats: NormalisationStats) -> np.ndarray
- fit_pca_with_observation(simulation_spectra: np.ndarray, observation_spectrum: np.ndarray, *, n_components: int = 2) -> PCAResults
- cumulative_explained_variance(pca_model: PCA) -> np.ndarray
- split_training_data(params, targets, *, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, random_state=42) -> DatasetSplits

American spellings (NormalizationStats, normalize_spectra, normalize_observation) are kept as compatibility aliases.

### Module: lba.emulator

#### Model and Config Classes

- NeuralNetworkEmulator(torch.nn.Module)
- TrainingConfig
- TrainingHistory
- EvaluationResult
- OptimisationResult

#### Public Functions

- build_emulator(*, input_dim=4, hidden_units=(128, 128, 32), dropout_rates=0.0, output_dim=2, device="cpu") -> NeuralNetworkEmulator
- train_emulator(model, x_train, y_train, *, x_val=None, y_val=None, config=TrainingConfig()) -> TrainingHistory
- predict(model, inputs, *, device="cpu") -> np.ndarray
- test_emulator(model, x_test, y_test, *, device="cpu") -> EvaluationResult
- optimise_emulator(...) -> OptimisationResult

The optimise_emulator routine supports:

- variable hidden-layer count
- variable hidden units per layer
- variable dropout rates per layer
- variable optimiser family
- variable learning rate
- early stopping per trial
- optional best-model checkpoint saving
- optional optimisation learning-curve plots

American alias optimize_emulator is also exported for backwards compatibility.

### Module: lba.cli

#### Public Functions

- build_parser() -> argparse.ArgumentParser
- run_pipeline(args: argparse.Namespace) -> dict[str, object]
- main() -> None

## CLI Parameters

Core:

- --data-dir
- --output-dir
- --n-components
- --train-fraction
- --val-fraction
- --test-fraction
- --epochs
- --validation-interval
- --learning-rate
- --hidden-units
- --device

Optimisation:

- --optimise (preferred)
- --n-trials
- --early-stopping-patience
- --early-stopping-min-delta
- --min-epochs-before-stopping
- --best-optimised-model-path
- --optimisation-curves-path
- --representative-trials

American compatibility aliases are accepted for optimisation flags.

Model saving:

- --save-model-path

## Output Artefacts

Typical outputs written to the output directory:

- train.npz, val.npz, test.npz
- observations_pca.npz
- pca_model.pkl
- emulator.pt

Optional optimisation outputs:

- best-optimised model checkpoint (custom path)
- optimisation training-curve plot (custom path)

## Testing

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run focused modules:

```bash
PYTHONPATH=src python3 -m unittest tests.test_emulator tests.test_cli -v
```
