from __future__ import annotations

"""Command-line interface for the end-to-end coursework pipeline.

Workflow:
1. Load observations and simulations.
2. Split into train/validation/test.
3. Compute normalisation statistics from training spectra only.
4. Apply those statistics to train/validation/test and observation.
5. Fit PCA on normalised training spectra (with observation appended).
6. Train and evaluate the emulator.
7. Optionally run hyperparameter optimisation.
"""

import argparse
import json
from pathlib import Path

import torch

from .emulator import TrainingConfig, build_emulator, optimise_emulator, test_emulator, train_emulator
from .io import (
    load_observations,
    load_simulation_dataset,
    save_observations_pca,
    save_pca_model,
    save_split_datasets,
)
from .preprocessing import fit_pca_with_observation, normalise_observation, normalise_spectra, split_training_data


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the APB86 A1 preprocessing and emulator training pipeline.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing observations.npz and simulations/")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where split datasets and trained artefacts will be written",
    )
    parser.add_argument("--n-components", type=int, default=2, help="Number of PCA components")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples for training")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples for validation")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of samples for testing")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10,
        help="How often to record validation loss during training",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimiser learning rate")
    parser.add_argument(
        "--hidden-units",
        type=int,
        nargs=3,
        default=(128, 128, 32),
        metavar=("H1", "H2", "H3"),
        help="Three hidden-layer widths for the emulator",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")

    parser.add_argument(
        "--optimise",
        action="store_true",
        dest="optimise",
        help="Run Optuna hyperparameter optimisation after training",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        dest="optimise",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=25,
        help="Patience in epochs for early stopping during each optimisation trial",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early-stopping patience",
    )
    parser.add_argument(
        "--min-epochs-before-stopping",
        type=int,
        default=25,
        help="Minimum epochs to run before early stopping can trigger",
    )

    parser.add_argument(
        "--best-optimised-model-path",
        default=None,
        help="Optional path to save the best model found during optimisation",
    )
    parser.add_argument("--best-optimized-model-path", default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--optimisation-curves-path",
        default=None,
        help="Optional path to save training-curve plots for optimisation trials",
    )
    parser.add_argument("--optimization-curves-path", default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--representative-trials",
        type=int,
        default=3,
        help="Number of non-best representative trials to include in optimisation plots",
    )
    parser.add_argument("--save-model-path", default=None, help="Optional explicit path for model checkpoint")
    return parser


def _pick_first(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Execute preprocessing, training, evaluation, and optional optimisation."""

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = load_observations(data_dir)
    simulations = load_simulation_dataset(data_dir)

    # Split first to prevent data leakage during normalisation.
    raw_splits = split_training_data(
        simulations.params,
        simulations.spectra,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    # Compute scaling stats from training spectra only, then reuse everywhere.
    normalised_train_spectra, normalisation_stats = normalise_spectra(raw_splits.y_train)
    normalised_val_spectra, _ = normalise_spectra(raw_splits.y_val, stats=normalisation_stats)
    normalised_test_spectra, _ = normalise_spectra(raw_splits.y_test, stats=normalisation_stats)
    normalised_observation = normalise_observation(observations.power, normalisation_stats)

    pca_results = fit_pca_with_observation(
        normalised_train_spectra,
        normalised_observation,
        n_components=args.n_components,
    )

    x_train = raw_splits.x_train
    x_val = raw_splits.x_val
    x_test = raw_splits.x_test
    y_train = pca_results.simulation_components
    y_val = pca_results.pca_model.transform(normalised_val_spectra)
    y_test = pca_results.pca_model.transform(normalised_test_spectra)

    save_split_datasets(
        output_dir,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    save_observations_pca(output_dir / "observations_pca.npz", pca_results.observation_components)
    save_pca_model(output_dir / "pca_model.pkl", pca_results.pca_model)

    config = TrainingConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_units=tuple(args.hidden_units),
        validation_interval=args.validation_interval,
        device=args.device,
    )

    model = build_emulator(
        input_dim=x_train.shape[1],
        hidden_units=config.hidden_units,
        output_dim=y_train.shape[1],
        device=config.device,
    )
    history = train_emulator(model, x_train, y_train, x_val=x_val, y_val=y_val, config=config)
    evaluation = test_emulator(model, x_test, y_test, device=config.device)

    model_path = Path(args.save_model_path) if args.save_model_path else output_dir / "emulator.pt"
    torch.save(model.state_dict(), model_path)

    result: dict[str, object] = {
        "num_samples": int(simulations.params.shape[0]),
        "num_features": int(simulations.spectra.shape[1]),
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
        "pca_components": int(args.n_components),
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
        "final_train_loss": float(history.train_loss[-1]),
        "final_val_loss": float(history.val_loss[-1]) if history.val_loss else None,
        "test_mse": float(evaluation.mse),
        "model_path": str(model_path),
        "normalisation_min_power": float(normalisation_stats.min_power),
        "normalisation_max_power": float(normalisation_stats.max_power),
    }

    if args.optimise:
        best_model_path = _pick_first(
            getattr(args, "best_optimised_model_path", None),
            getattr(args, "best_optimized_model_path", None),
        )
        curves_path = _pick_first(
            getattr(args, "optimisation_curves_path", None),
            getattr(args, "optimization_curves_path", None),
        )

        optimisation_result = optimise_emulator(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=x_train.shape[1],
            output_dim=y_train.shape[1],
            epochs=args.epochs,
            n_trials=args.n_trials,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            min_epochs_before_stopping=args.min_epochs_before_stopping,
            best_model_path=best_model_path,
            training_curves_plot_path=curves_path,
            representative_trial_count=args.representative_trials,
            device=args.device,
        )
        result["best_hyperparameters"] = optimisation_result.best_params
        result["best_validation_loss"] = optimisation_result.best_value
        result["best_optimised_model_path"] = optimisation_result.best_model_path
        result["optimisation_curves_plot_path"] = optimisation_result.training_curves_plot_path
        # Backwards-compatible American-English output keys.
        result["best_optimized_model_path"] = optimisation_result.best_model_path
        result["optimization_curves_plot_path"] = optimisation_result.training_curves_plot_path

    return result


def main() -> None:
    """CLI entrypoint used by both ``python -m apb86_a1`` and console script."""

    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
