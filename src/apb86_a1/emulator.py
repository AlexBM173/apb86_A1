from __future__ import annotations

"""Neural-network emulator utilities for training, evaluation, and optimization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


class NeuralNetworkEmulator(torch.nn.Module):
    """Fully connected emulator network with optional per-layer dropout.

    Args:
        input_dim: Number of input features.
        hidden_units: Hidden-layer widths in order.
        dropout_rates: Scalar or per-layer dropout values in `[0, 1)`.
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
        normalized_dropout_rates = _normalize_dropout_rates(hidden_units, dropout_rates)
        for hidden_dim, dropout_rate in zip(hidden_units, normalized_dropout_rates):
            layers.append(torch.nn.Linear(current_dim, int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(p=float(dropout_rate)))
            current_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass for a batch of inputs."""

        return self.network(x)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for deterministic emulator training.

    Attributes:
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        hidden_units: Hidden-layer architecture.
        dropout_rates: Per-layer dropout rates used during model construction.
        validation_interval: Epoch interval for validation-loss recording.
        optimizer_name: Optimizer identifier (e.g., `adam`, `sgd`).
        device: Torch device string.
    """

    learning_rate: float = 1e-3
    epochs: int = 200
    hidden_units: tuple[int, ...] = (128, 128, 32)
    dropout_rates: tuple[float, ...] = (0.0, 0.0, 0.0)
    validation_interval: int = 10
    optimizer_name: str = "adam"
    device: str = "cpu"


@dataclass
class TrainingHistory:
    """Training and validation losses recorded during training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outputs for test-time assessment."""

    mse: float
    predictions: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    """Summary outputs from Optuna hyperparameter optimization."""

    best_params: dict[str, Any]
    best_value: float
    study: object
    best_model_path: str | None = None
    training_curves_plot_path: str | None = None


def _normalize_dropout_rates(
    hidden_units: Sequence[int],
    dropout_rates: float | Sequence[float],
) -> tuple[float, ...]:
    """Normalize scalar or sequence dropout settings to per-layer tuple.

    Args:
        hidden_units: Hidden architecture used for length validation.
        dropout_rates: Scalar or per-layer dropout specification.

    Returns:
        Per-layer dropout-rate tuple.
    """

    if isinstance(dropout_rates, (int, float)):
        return tuple(float(dropout_rates) for _ in hidden_units)

    normalized_dropout_rates = tuple(float(rate) for rate in dropout_rates)
    if len(normalized_dropout_rates) != len(hidden_units):
        raise ValueError("dropout_rates must match the number of hidden layers")
    for rate in normalized_dropout_rates:
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout rates must be in the range [0, 1)")
    return normalized_dropout_rates


def _build_optimizer(
    parameters: Any,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Construct a torch optimizer by name.

    Args:
        parameters: Model parameters iterable.
        optimizer_name: Optimizer identifier.
        learning_rate: Learning rate value.

    Returns:
        Configured torch optimizer instance.
    """

    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if normalized_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if normalized_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if normalized_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    """Convert numpy arrays to float32 tensors on the requested device."""

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
    """Train an emulator model and return per-epoch loss history.

    Args:
        model: Emulator model to train.
        x_train: Training features.
        y_train: Training targets.
        x_val: Optional validation features.
        y_val: Optional validation targets.
        config: Training configuration.

    Returns:
        Training history including train and optionally validation losses.
    """

    x_train_tensor = _to_tensor(x_train, config.device)
    y_train_tensor = _to_tensor(y_train, config.device)
    x_val_tensor = _to_tensor(x_val, config.device) if x_val is not None else None
    y_val_tensor = _to_tensor(y_val, config.device) if y_val is not None else None

    criterion = torch.nn.MSELoss()
    optimizer = _build_optimizer(
        model.parameters(),
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
    )
    history = TrainingHistory()

    for epoch in range(config.epochs):
        # Standard supervised update: forward -> loss -> backward -> step.
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        history.train_loss.append(float(loss.item()))

        should_validate = (
            x_val_tensor is not None
            and y_val_tensor is not None
            and (epoch + 1) % config.validation_interval == 0
        )
        if should_validate:
            # Validation is performed in eval mode and without gradient tracking
            # to keep metrics unbiased and memory usage low.
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            history.val_loss.append(float(val_loss.item()))

    return history


def predict(model: NeuralNetworkEmulator, inputs: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Generate emulator predictions for numpy inputs.

    Args:
        model: Trained emulator model.
        inputs: Feature matrix to predict on.
        device: Device used for tensor execution.

    Returns:
        Predicted outputs as a numpy array.
    """

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
    """Evaluate emulator predictions on test data with MSE metric.

    Args:
        model: Trained emulator model.
        x_test: Test features.
        y_test: Test targets.
        device: Device used for tensor execution.

    Returns:
        Evaluation result containing MSE, predictions, and targets.
    """

    predictions = predict(model, x_test, device=device)
    targets = np.asarray(y_test, dtype=float)
    mse = float(np.mean((predictions - targets) ** 2))
    return EvaluationResult(mse=mse, predictions=predictions, targets=targets)


def optimize_emulator(
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
) -> OptimizationResult:
    """Run Optuna optimization over architecture and training hyperparameters.

    The search space includes variable layer counts, per-layer hidden widths,
    per-layer dropout rates, optimizer selection, and learning rate. Each trial
    supports early stopping and records train/validation curves for later
    visualization.

    Args:
        x_train: Training features.
        y_train: Training targets.
        x_val: Validation features.
        y_val: Validation targets.
        input_dim: Input dimensionality.
        output_dim: Output dimensionality.
        epochs: Maximum epochs per trial.
        n_trials: Number of Optuna trials.
        layer_options: Candidate values for hidden-layer count.
        hidden_unit_options: Candidate hidden-layer widths.
        dropout_range: Continuous range for dropout-rate sampling.
        optimizer_options: Candidate optimizer names.
        learning_rate_range: Continuous range for learning-rate sampling.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_min_delta: Minimum improvement threshold.
        min_epochs_before_stopping: Warmup epochs before early stopping allowed.
        best_model_path: Optional path to persist best model checkpoint.
        training_curves_plot_path: Optional path for optimization-curve plot.
        representative_trial_count: Number of non-best trials to plot.
        device: Torch device for optimization runs.

    Returns:
        Optimization result with best params/value and optional artifact paths.
    """

    if optuna is None:
        raise ImportError("optuna is required to optimize emulator hyperparameters")

    x_train_tensor = _to_tensor(x_train, device)
    y_train_tensor = _to_tensor(y_train, device)
    x_val_tensor = _to_tensor(x_val, device)
    y_val_tensor = _to_tensor(y_val, device)
    criterion = torch.nn.MSELoss()

    def _select_representative_trial_numbers() -> list[int]:
        """Choose non-best trials that span validation-loss quantiles."""

        completed = [
            trial
            for trial in study.trials
            if trial.value is not None and trial.number != study.best_trial.number
        ]
        if not completed:
            return []
        completed_sorted = sorted(completed, key=lambda trial: float(trial.value))
        target_count = min(representative_trial_count, len(completed_sorted))
        if target_count == len(completed_sorted):
            return [trial.number for trial in completed_sorted]
        quantile_indices = np.linspace(0, len(completed_sorted) - 1, target_count, dtype=int)
        return [completed_sorted[index].number for index in quantile_indices]

    def _save_training_curves(plot_path: Path) -> None:
        """Save optimization learning-curve visualizations to disk."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "matplotlib is required to save optimization training curves"
            ) from exc

        best_trial = study.best_trial
        best_train = best_trial.user_attrs.get("train_loss_history", [])
        best_val = best_trial.user_attrs.get("val_loss_history", [])
        representative_numbers = _select_representative_trial_numbers()

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
            trial = study.trials[trial_number]
            val_history = trial.user_attrs.get("val_loss_history", [])
            axes[1].plot(
                val_history,
                linestyle="--",
                alpha=0.8,
                label=f"Trial #{trial.number}",
            )
        axes[1].set_title("Validation Curves: Best + Representative Trials")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: train one sampled configuration and return best val loss."""

        n_hidden_layers = trial.suggest_categorical("n_hidden_layers", list(layer_options))
        hidden_units = tuple(
            trial.suggest_categorical(
                f"n_hidden_units_{layer_index + 1}",
                list(hidden_unit_options),
            )
            for layer_index in range(n_hidden_layers)
        )
        dropout_rates = tuple(
            trial.suggest_float(
                f"dropout_rate_{layer_index + 1}",
                dropout_range[0],
                dropout_range[1],
            )
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
        optimizer = _build_optimizer(
            model.parameters(),
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
        )

        train_loss_history: list[float] = []
        val_loss_history: list[float] = []
        best_val_loss = float("inf")
        epochs_since_improvement = 0

        for epoch in range(epochs):
            # Single-epoch training update.
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()

            train_loss_history.append(float(train_loss.item()))

            # Evaluate on validation split every epoch for reliable early stopping
            # and richer curve reporting.
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
                raise optuna.TrialPruned()

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

    # Minimize validation MSE across sampled hyperparameters.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_model_path_str: str | None = None
    if best_model_path is not None:
        # Reconstruct and retrain the best configuration for exactly the number
        # of epochs executed in its winning trial so the saved checkpoint mirrors
        # the validated trial behavior.
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
        best_optimizer = _build_optimizer(
            best_model.parameters(),
            optimizer_name=str(best_trial.params["optimizer_name"]),
            learning_rate=float(best_trial.params["learning_rate"]),
        )
        for _ in range(int(best_trial.user_attrs.get("epochs_run", epochs))):
            best_model.train()
            best_optimizer.zero_grad()
            outputs = best_model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            best_optimizer.step()
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in best_model.state_dict().items()
        }

        best_model_path_obj = Path(best_model_path)
        best_model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, best_model_path_obj)
        best_model_path_str = str(best_model_path_obj)

    training_curves_plot_path_str: str | None = None
    if training_curves_plot_path is not None:
        plot_path_obj = Path(training_curves_plot_path)
        _save_training_curves(plot_path_obj)
        training_curves_plot_path_str = str(plot_path_obj)

    return OptimizationResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        study=study,
        best_model_path=best_model_path_str,
        training_curves_plot_path=training_curves_plot_path_str,
    )