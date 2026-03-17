from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from apb86_a1.emulator import (
    NeuralNetworkEmulator,
    build_emulator,
    optimize_emulator,
    predict,
    test_emulator,
    optuna,
)


class EmulatorTests(unittest.TestCase):
    def test_build_emulator_supports_variable_layers_and_dropout(self) -> None:
        model = build_emulator(
            input_dim=4,
            hidden_units=(16, 8),
            dropout_rates=(0.25, 0.1),
            output_dim=2,
        )

        self.assertIsInstance(model, NeuralNetworkEmulator)
        dropout_layers = [module for module in model.network if isinstance(module, torch.nn.Dropout)]
        self.assertEqual(len(dropout_layers), 2)

    def test_predict_returns_expected_shape(self) -> None:
        model = build_emulator(input_dim=4, hidden_units=(8, 4, 2), output_dim=2)
        inputs = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=float)

        predictions = predict(model, inputs)

        self.assertEqual(predictions.shape, (2, 2))

    def test_test_emulator_returns_zero_mse_for_zero_targets(self) -> None:
        model = build_emulator(input_dim=4, hidden_units=(4, 4, 4), output_dim=2)
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)

        x_test = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=float)
        y_test = np.zeros((2, 2), dtype=float)

        evaluation = test_emulator(model, x_test, y_test)

        self.assertEqual(evaluation.predictions.shape, (2, 2))
        self.assertEqual(evaluation.targets.shape, (2, 2))
        self.assertAlmostEqual(evaluation.mse, 0.0)

    @unittest.skipIf(optuna is None, "optuna is not installed")
    def test_optimize_emulator_returns_best_params(self) -> None:
        x_train = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        y_train = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        x_val = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=float)
        y_val = np.array([[0.5, 0.5]], dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            best_model_path = Path(tmpdir) / "best_optimized_model.pt"
            curves_path = Path(tmpdir) / "optimization_curves.png"

            result = optimize_emulator(
                x_train,
                y_train,
                x_val,
                y_val,
                input_dim=4,
                output_dim=2,
                epochs=10,
                n_trials=2,
                layer_options=(1, 2, 3),
                hidden_unit_options=(4, 8),
                dropout_range=(0.0, 0.3),
                optimizer_options=("adam", "sgd"),
                learning_rate_range=(1e-4, 1e-2),
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                min_epochs_before_stopping=2,
                best_model_path=best_model_path,
                training_curves_plot_path=curves_path,
                representative_trial_count=1,
            )

            self.assertIn("n_hidden_layers", result.best_params)
            self.assertIn("optimizer_name", result.best_params)
            self.assertIn("learning_rate", result.best_params)
            self.assertIn("n_hidden_units_1", result.best_params)
            self.assertIn("dropout_rate_1", result.best_params)
            self.assertIsInstance(result.best_value, float)
            self.assertEqual(len(result.study.trials), 2)
            self.assertIsNotNone(result.best_model_path)
            self.assertIsNotNone(result.training_curves_plot_path)
            self.assertTrue(best_model_path.exists())
            self.assertTrue(curves_path.exists())


if __name__ == "__main__":
    unittest.main()