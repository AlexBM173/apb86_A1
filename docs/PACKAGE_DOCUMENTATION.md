# APB86 A1 Package Documentation

## Overview

The `apb86_a1` package provides a complete workflow for:

- loading 21-cm observation and simulation data
- normalizing spectra and applying PCA
- splitting datasets for machine-learning workflows
- training and evaluating a neural-network emulator
- running Optuna-based hyperparameter optimization with:
  - variable depth/width architectures
  - dropout search
  - optimizer and learning-rate search
  - early stopping
  - best-model checkpointing
  - optimization curve visualization

## Installation and Execution

Use directly from the repository:

```bash
PYTHONPATH=src python3 -m apb86_a1 --output-dir data
```

Or after installation:

```bash
apb86-a1 --output-dir data
```

## Package Layout

- `src/apb86_a1/io.py`: data loading and persistence helpers
- `src/apb86_a1/preprocessing.py`: scaling, PCA, and data splitting
- `src/apb86_a1/emulator.py`: model, training, evaluation, optimization
- `src/apb86_a1/cli.py`: end-to-end command-line pipeline
- `src/apb86_a1/__init__.py`: public API exports

## API Reference

### Module: `apb86_a1.io`

#### Data Classes

- `ObservationData`
  - `k: np.ndarray`
  - `power: np.ndarray`

- `SimulationData`
  - `ks: np.ndarray`
  - `spectra: np.ndarray`
  - `params: np.ndarray`
  - `redshift: np.ndarray`
  - `code: list[str]`
  - `code_version: list[str]`
  - `filenames: list[str]`

#### Public Functions

- `load_observations(data_dir: str | Path) -> ObservationData`
  - Loads `observations.npz`.

- `load_simulation_dataset(data_dir: str | Path) -> SimulationData`
  - Loads all simulation files and flattens parameters into 4-value vectors.

- `save_split_datasets(output_dir, *, x_train, y_train, x_val, y_val, x_test, y_test) -> None`
  - Saves train/val/test arrays as `.npz` files.

- `load_split_dataset(file_path: str | Path) -> dict[str, np.ndarray]`
  - Reads one saved split file.

- `save_observations_pca(output_path: str | Path, observations_pca: np.ndarray) -> None`
  - Saves PCA-transformed observation data.

- `save_pca_model(output_path: str | Path, pca_model: Any) -> None`
  - Saves a fitted PCA model using joblib.

### Module: `apb86_a1.preprocessing`

#### Data Classes

- `NormalizationStats`
  - `min_power: float`
  - `max_power: float`
  - `scale: float` property

- `PCAResults`
  - `pca_model: sklearn.decomposition.PCA`
  - `simulation_components: np.ndarray`
  - `observation_components: np.ndarray`

- `DatasetSplits`
  - `x_train, x_val, x_test, y_train, y_val, y_test`

#### Public Functions

- `normalize_spectra(spectra: np.ndarray, stats: NormalizationStats | None = None) -> tuple[np.ndarray, NormalizationStats]`
  - Min-max normalizes simulation spectra.

- `normalize_observation(observation_spectrum: np.ndarray, stats: NormalizationStats) -> np.ndarray`
  - Normalizes observation with simulation-derived stats.

- `fit_pca_with_observation(simulation_spectra: np.ndarray, observation_spectrum: np.ndarray, *, n_components: int = 2) -> PCAResults`
  - Fits PCA using simulations + observation, transforms both.

- `cumulative_explained_variance(pca_model: PCA) -> np.ndarray`
  - Returns cumulative explained variance.

- `split_training_data(params, targets, *, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, random_state=42) -> DatasetSplits`
  - Produces train/validation/test split arrays.

### Module: `apb86_a1.emulator`

#### Model and Config Classes

- `NeuralNetworkEmulator(torch.nn.Module)`
  - Configurable dense network with optional per-layer dropout.

- `TrainingConfig`
  - `learning_rate`, `epochs`, `hidden_units`, `dropout_rates`, `validation_interval`, `optimizer_name`, `device`

- `TrainingHistory`
  - `train_loss: list[float]`
  - `val_loss: list[float]`

- `EvaluationResult`
  - `mse`, `predictions`, `targets`

- `OptimizationResult`
  - `best_params`, `best_value`, `study`, `best_model_path`, `training_curves_plot_path`

#### Public Functions

- `build_emulator(*, input_dim=4, hidden_units=(128, 128, 32), dropout_rates=0.0, output_dim=2, device="cpu") -> NeuralNetworkEmulator`
  - Builds and moves model to target device.

- `train_emulator(model, x_train, y_train, *, x_val=None, y_val=None, config=TrainingConfig()) -> TrainingHistory`
  - Trains model and records losses.

- `predict(model, inputs, *, device="cpu") -> np.ndarray`
  - Performs inference.

- `test_emulator(model, x_test, y_test, *, device="cpu") -> EvaluationResult`
  - Computes predictions and test MSE.

- `optimize_emulator(...) -> OptimizationResult`
  - Runs Optuna optimization with support for:
    - variable hidden-layer count
    - variable hidden units per layer
    - variable dropout rates per layer
    - variable optimizer family
    - variable learning rate
    - early stopping per trial
    - best-model checkpoint saving
    - optimization curve plotting

### Module: `apb86_a1.cli`

#### Public Functions

- `build_parser() -> argparse.ArgumentParser`
  - Creates the pipeline CLI parser.

- `run_pipeline(args: argparse.Namespace) -> dict[str, object]`
  - Executes full workflow and returns JSON-serializable summary.

- `main() -> None`
  - Entry point used by console script and `python -m apb86_a1`.

## CLI Parameters

Core:

- `--data-dir`
- `--output-dir`
- `--n-components`
- `--train-fraction`
- `--val-fraction`
- `--test-fraction`
- `--epochs`
- `--validation-interval`
- `--learning-rate`
- `--hidden-units`
- `--device`

Optimization:

- `--optimize`
- `--n-trials`
- `--early-stopping-patience`
- `--early-stopping-min-delta`
- `--min-epochs-before-stopping`
- `--best-optimized-model-path`
- `--optimization-curves-path`
- `--representative-trials`

Model saving:

- `--save-model-path`

## Output Artifacts

Typical outputs written to output directory:

- `train.npz`, `val.npz`, `test.npz`
- `observations_pca.npz`
- `pca_model.pkl`
- `emulator.pt`

Optional optimization outputs:

- best-optimized model checkpoint (custom path)
- optimization training-curve plot (custom path)

## Testing

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run focused modules:

```bash
PYTHONPATH=src python3 -m unittest tests.test_emulator tests.test_cli -v
```
