# apb86_A1
A1 Radio Astronomy coursework for Cambridge MPhil in Data Intensive Science

## Python Library

The project now includes a small Python package under `src/lba` for:

- loading the observation and simulation datasets
- normalising spectra and applying PCA
- training, evaluating, and optimising the neural network emulator

The package normalisation is leakage-safe: it splits simulation data first, computes
min/max power from the training spectra only, then applies those statistics to
validation, test, and observation spectra.

Full API and usage documentation is available in `docs/PACKAGE_DOCUMENTATION.md`.

Run the package without installing it by setting `PYTHONPATH=src`.

## CLI Usage

Run the full preprocessing and emulator pipeline from the command line:

```bash
PYTHONPATH=src python3 -m lba --output-dir data
```

If the package is installed, the console script is also available:

```bash
lba --output-dir data
```

Useful options:

```bash
PYTHONPATH=src python3 -m lba \
	--output-dir data \
	--epochs 200 \
	--validation-interval 10 \
	--n-components 3 \
	--train-fraction 0.7 \
	--val-fraction 0.2 \
	--test-fraction 0.1 \
	--hidden-units 128 128 32
```

To run hyperparameter optimisation:

```bash
PYTHONPATH=src python3 -m lba \
	--output-dir data \
	--optimise \
	--n-trials 20 \
	--best-optimised-model-path data/best_optimised_emulator.pt \
	--optimisation-curves-path data/optimisation_curves.png
```

The command writes these artifacts to the output directory:

- `train.npz`, `val.npz`, `test.npz`
- `observations_pca.npz`
- `pca_model.pkl`
- `emulator.pt`

Example JSON output:

```json
{
	"final_train_loss": 0.10465678572654724,
	"final_val_loss": 0.09140075743198395,
	"model_path": "/tmp/example/emulator.pt",
	"num_features": 54,
	"num_samples": 9997,
	"pca_components": 3,
	"test_fraction": 0.1,
	"test_mse": 0.1009839330711191,
	"test_size": 1000,
	"train_fraction": 0.7,
	"train_size": 6997,
	"val_fraction": 0.2,
	"val_size": 2000
}
```
