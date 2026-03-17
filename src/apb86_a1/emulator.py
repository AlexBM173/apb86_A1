from __future__ import annotations

"""Neural-network emulator utilities for training, evaluation, and optimisation.

This module provides fully connected PyTorch neural-network models for emulating
relationships between astrophysical parameters and PCA-compressed spectra.
It supports variable architecture search, early stopping, and Optuna-based
hyperparameter optimisation with trial-curve visualisation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


class NeuralNetworkEmulator(torch.nn.Module):
    """Fully connected emulator with optional per-layer dropout.

    Args:
        input_dim: Number of input features.
        hidden_units: Hidden-layer widths in order.
        dropout_rates: Scalar or per-layer dropout values in ``[0, 1)``.
        output_dim: Number of output targets.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_units: Sequence[int] = (128, 128, 32),
        dropout_rates: float | Sequence[float] = 0.0,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        if len(hidden_units) == 0:
            raise ValueError("hidden_units must contain at least one layer size")

        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        normalised_dropout_rates = _normalise_dropout_rates(hidden_units, dropout_rates)

        for hidden_dim, dropout_rate in zip(hidden_units, normalised_dropout_rates):
            layers.append(torch.nn.Linear(current_dim, int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(p=float(dropout_rate)))
            current_dim = int(hidden_dim)

        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass for a batch of inputs."""

        return self.network(x)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for deterministic emulator training."""

    learning_rate: float = 1e-3
    epochs: int = 200
    hidden_units: tuple[int, ...] = (128, 128, 32)
    dropout_rates: tuple[float, ...] = (0.0, 0.0, 0.0)
    validation_interval: int = 10
    optimizer_name: str = "adam"
    device: str = "cpu"


@dataclass
class TrainingHistory:
    """Per-epoch train and validation losses."""

    train_loss: list[float] = field(default_factory=lambda: cast(list[float], []))
    val_loss: list[float] = field(default_factory=lambda: cast(list[float], []))


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outputs for held-out test data."""

    mse: float
    predictions: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class OptimisationResult:
    """Summary outputs from Optuna hyperparameter optimisation."""

    best_params: dict[str, Any]
    best_value: float
    study: object
    best_model_path: str | None = None
    training_curves_plot_path: str | None = None


# Backwards-compatibility alias for American English spelling.
OptimizationResult = OptimisationResult


def _normalise_dropout_rates(
    hidden_units: Sequence[int],
    dropout_rates: float | Sequence[float],
) -> tuple[float, ...]:
    """Normalise scalar or sequence dropout settings to a per-layer tuple."""

    if isinstance(dropout_rates, (int, float)):
        rate = float(dropout_rates)
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout rates must be in the range [0, 1)")
        return tuple(rate for _ in hidden_units)

    normalised_dropout_rates = tuple(float(rate) for rate in dropout_rates)
    if len(normalised_dropout_rates) != len(hidden_units):
        raise ValueError("dropout_rates must match the number of hidden layers")
    for rate in normalised_dropout_rates:
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout rates must be in the range [0, 1)")
    return normalised_dropout_rates


def _build_optimizer(
    parameters: Any,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Construct a torch optimiser by name."""

    normalised_name = optimizer_name.lower()
    if normalised_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if normalised_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if normalised_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if normalised_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimiser: {optimizer_name}")


def _to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    """Convert NumPy arrays to float32 torch tensors on ``device``."""

    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)


def build_emulator(
    *,
    input_dim: int = 4,
    hidden_units: Sequence[int] = (128, 128, 32),
    dropout_rates: float | Sequence[float] = 0.0,
    output_dim: int = 2,
    device: str = "cpu",
) -> NeuralNetworkEmulator:
    """Build and place a neural-network emulator on the selected device."""

    model = NeuralNetworkEmulator(
        input_dim=input_dim,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        output_dim=output_dim,
    )
    return model.to(device)


def train_emulator(
    model: NeuralNetworkEmulator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    config: TrainingConfig = TrainingConfig(),
) -> TrainingHistory:
    """Train an emulator model and return loss history."""

    x_train_tensor = _to_tensor(x_train, config.device)
    y_train_tensor = _to_tensor(y_train, config.device)
    x_val_tensor = _to_tensor(x_val, config.device) if x_val is not None else None
    y_val_tensor = _to_tensor(y_val, config.device) if y_val is not None else None

    criterion = torch.nn.MSELoss()
    optimiser = _build_optimizer(
        model.parameters(),
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
    )

    history = TrainingHistory()

    for epoch in range(config.epochs):
        model.train()
        optimiser.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimiser.step()
        history.train_loss.append(float(loss.item()))

        should_validate = (
            x_val_tensor is not None
            and y_val_tensor is not None
            and (epoch + 1) % config.validation_interval == 0
        )
        if should_validate:
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            history.val_loss.append(float(val_loss.item()))

    return history


def predict(model: NeuralNetworkEmulator, inputs: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Generate emulator predictions for NumPy inputs."""

    model.eval()
    with torch.no_grad():
        outputs = model(_to_tensor(inputs, device))
    return outputs.detach().cpu().numpy()


def test_emulator(
    model: NeuralNetworkEmulator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    device: str = "cpu",
) -> EvaluationResult:
    """Evaluate emulator predictions on test data with MSE."""

    predictions = predict(model, x_test, device=device)
    targets = np.asarray(y_test, dtype=float)
    mse = float(np.mean((predictions - targets) ** 2))
    return EvaluationResult(mse=mse, predictions=predictions, targets=targets)


def optimise_emulator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int = 4,
    output_dim: int = 2,
    epochs: int = 200,
    n_trials: int = 20,
    layer_options: Sequence[int] = (1, 2, 3, 4),
    hidden_unit_options: Sequence[int] = (16, 32, 64, 128, 256),
    dropout_range: tuple[float, float] = (0.0, 0.5),
    optimizer_options: Sequence[str] = ("adam", "adamw", "rmsprop", "sgd"),
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2),
    early_stopping_patience: int = 25,
    early_stopping_min_delta: float = 0.0,
    min_epochs_before_stopping: int = 25,
    best_model_path: str | Path | None = None,
    training_curves_plot_path: str | Path | None = None,
    representative_trial_count: int = 3,
    device: str = "cpu",
) -> OptimisationResult:
    """Run Optuna hyperparameter optimisation over architecture and training settings."""

    if optuna is None:
        raise ImportError("optuna is required to optimise emulator hyperparameters")
    assert optuna is not None
    optuna_module = optuna

    x_train_tensor = _to_tensor(x_train, device)
    y_train_tensor = _to_tensor(y_train, device)
    x_val_tensor = _to_tensor(x_val, device)
    y_val_tensor = _to_tensor(y_val, device)
    criterion = torch.nn.MSELoss()

    def _select_representative_trial_numbers(study_obj: Any) -> list[int]:
        completed = [
            trial
            for trial in study_obj.trials
            if trial.value is not None and trial.number != study_obj.best_trial.number
        ]
        if not completed:
            return []
        completed_sorted = sorted(completed, key=lambda trial: float(trial.value))
        target_count = min(representative_trial_count, len(completed_sorted))
        if target_count == len(completed_sorted):
            return [trial.number for trial in completed_sorted]
        quantile_indices = np.linspace(0, len(completed_sorted) - 1, target_count, dtype=int)
        return [completed_sorted[index].number for index in quantile_indices]

    def _save_training_curves(study_obj: Any, plot_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required to save optimisation training curves") from exc

        plt = cast(Any, plt)

        best_trial = study_obj.best_trial
        best_train = best_trial.user_attrs.get("train_loss_history", [])
        best_val = best_trial.user_attrs.get("val_loss_history", [])
        representative_numbers = _select_representative_trial_numbers(study_obj)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        axes[0].plot(best_train, label="Best Trial Train", color="tab:blue", linewidth=2)
        axes[0].plot(best_val, label="Best Trial Val", color="tab:orange", linewidth=2)
        axes[0].set_title("Best Trial Learning Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(best_val, label=f"Best Trial (#{best_trial.number})", color="black", linewidth=2)
        for trial_number in representative_numbers:
            trial = study_obj.trials[trial_number]
            val_history = trial.user_attrs.get("val_loss_history", [])
            axes[1].plot(val_history, linestyle="--", alpha=0.8, label=f"Trial #{trial.number}")

        axes[1].set_title("Validation Curves: Best + Representative Trials")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def objective(trial: Any) -> float:
        n_hidden_layers = trial.suggest_categorical("n_hidden_layers", list(layer_options))
        hidden_units = tuple(
            trial.suggest_categorical(f"n_hidden_units_{layer_index + 1}", list(hidden_unit_options))
            for layer_index in range(n_hidden_layers)
        )
        dropout_rates = tuple(
            trial.suggest_float(f"dropout_rate_{layer_index + 1}", dropout_range[0], dropout_range[1])
            for layer_index in range(n_hidden_layers)
        )
        optimizer_name = trial.suggest_categorical("optimizer_name", list(optimizer_options))
        learning_rate = trial.suggest_float(
            "learning_rate",
            learning_rate_range[0],
            learning_rate_range[1],
            log=True,
        )

        model = build_emulator(
            input_dim=input_dim,
            hidden_units=hidden_units,
            dropout_rates=dropout_rates,
            output_dim=output_dim,
            device=device,
        )
        optimiser = _build_optimizer(model.parameters(), optimizer_name, learning_rate)

        train_loss_history: list[float] = []
        val_loss_history: list[float] = []
        best_val_loss = float("inf")
        epochs_since_improvement = 0

        for epoch in range(epochs):
            model.train()
            optimiser.zero_grad()
            outputs = model(x_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimiser.step()
            train_loss_history.append(float(train_loss.item()))

            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            current_val_loss = float(val_loss.item())
            val_loss_history.append(current_val_loss)

            if current_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = current_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            trial.report(current_val_loss, step=epoch)
            if trial.should_prune():
                raise optuna_module.TrialPruned()

            can_stop_early = (
                early_stopping_patience > 0
                and (epoch + 1) >= min_epochs_before_stopping
                and epochs_since_improvement >= early_stopping_patience
            )
            if can_stop_early:
                break

        trial.set_user_attr("train_loss_history", train_loss_history)
        trial.set_user_attr("val_loss_history", val_loss_history)
        trial.set_user_attr("epochs_run", len(train_loss_history))
        return best_val_loss

    study = optuna_module.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_model_path_str: str | None = None
    if best_model_path is not None:
        best_trial = study.best_trial
        best_model = build_emulator(
            input_dim=input_dim,
            hidden_units=tuple(
                int(best_trial.params[f"n_hidden_units_{layer_idx + 1}"])
                for layer_idx in range(int(best_trial.params["n_hidden_layers"]))
            ),
            dropout_rates=tuple(
                float(best_trial.params[f"dropout_rate_{layer_idx + 1}"])
                for layer_idx in range(int(best_trial.params["n_hidden_layers"]))
            ),
            output_dim=output_dim,
            device=device,
        )
        best_optimiser = _build_optimizer(
            best_model.parameters(),
            optimizer_name=str(best_trial.params["optimizer_name"]),
            learning_rate=float(best_trial.params["learning_rate"]),
        )

        for _ in range(int(best_trial.user_attrs.get("epochs_run", epochs))):
            best_model.train()
            best_optimiser.zero_grad()
            outputs = best_model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            best_optimiser.step()

        best_state_dict = {key: value.detach().cpu().clone() for key, value in best_model.state_dict().items()}
        best_model_path_obj = Path(best_model_path)
        best_model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, best_model_path_obj)
        best_model_path_str = str(best_model_path_obj)

    training_curves_plot_path_str: str | None = None
    if training_curves_plot_path is not None:
        plot_path_obj = Path(training_curves_plot_path)
        _save_training_curves(study, plot_path_obj)
        training_curves_plot_path_str = str(plot_path_obj)

    return OptimisationResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        study=study,
        best_model_path=best_model_path_str,
        training_curves_plot_path=training_curves_plot_path_str,
    )


# Backwards-compatibility alias for American English spelling.
def optimize_emulator(*args: Any, **kwargs: Any) -> OptimisationResult:
    """Deprecated alias for :func:`optimise_emulator`."""

    return optimise_emulator(*args, **kwargs)
