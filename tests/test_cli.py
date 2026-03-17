from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from apb86_a1.cli import build_parser, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class CLITests(unittest.TestCase):
    def test_build_parser_accepts_custom_split_fractions(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--train-fraction",
                "0.7",
                "--val-fraction",
                "0.2",
                "--test-fraction",
                "0.1",
                "--n-components",
                "3",
            ]
        )

        self.assertEqual(args.train_fraction, 0.7)
        self.assertEqual(args.val_fraction, 0.2)
        self.assertEqual(args.test_fraction, 0.1)
        self.assertEqual(args.n_components, 3)

    def test_run_pipeline_writes_artifacts_and_honors_custom_settings(self) -> None:
        parser = build_parser()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(
                [
                    "--data-dir",
                    str(DATA_DIR),
                    "--output-dir",
                    tmpdir,
                    "--epochs",
                    "1",
                    "--validation-interval",
                    "1",
                    "--n-components",
                    "3",
                    "--train-fraction",
                    "0.7",
                    "--val-fraction",
                    "0.2",
                    "--test-fraction",
                    "0.1",
                ]
            )

            result = run_pipeline(args)
            output_dir = Path(tmpdir)

            self.assertEqual(result["pca_components"], 3)
            self.assertEqual(result["train_fraction"], 0.7)
            self.assertEqual(result["val_fraction"], 0.2)
            self.assertEqual(result["test_fraction"], 0.1)
            self.assertEqual(result["num_samples"], 9997)
            self.assertEqual(result["train_size"], 6997)
            self.assertEqual(result["val_size"], 2000)
            self.assertEqual(result["test_size"], 1000)
            self.assertEqual(result["train_size"] + result["val_size"] + result["test_size"], result["num_samples"])
            self.assertTrue((output_dir / "train.npz").exists())
            self.assertTrue((output_dir / "val.npz").exists())
            self.assertTrue((output_dir / "test.npz").exists())
            self.assertTrue((output_dir / "observations_pca.npz").exists())
            self.assertTrue((output_dir / "pca_model.pkl").exists())
            self.assertTrue((output_dir / "emulator.pt").exists())

            observations_pca = np.load(output_dir / "observations_pca.npz")
            self.assertEqual(observations_pca["observations_pca"].shape, (1, 3))

    def test_run_pipeline_optimization_outputs_best_model_and_plot(self) -> None:
        parser = build_parser()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            best_model_path = output_dir / "best_optimized_model.pt"
            curves_path = output_dir / "optimization_curves.png"

            args = parser.parse_args(
                [
                    "--data-dir",
                    str(DATA_DIR),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "3",
                    "--validation-interval",
                    "1",
                    "--optimize",
                    "--n-trials",
                    "1",
                    "--early-stopping-patience",
                    "1",
                    "--min-epochs-before-stopping",
                    "1",
                    "--best-optimized-model-path",
                    str(best_model_path),
                    "--optimization-curves-path",
                    str(curves_path),
                    "--representative-trials",
                    "1",
                ]
            )

            result = run_pipeline(args)

            self.assertIn("best_hyperparameters", result)
            self.assertIn("best_validation_loss", result)
            self.assertEqual(result["best_optimized_model_path"], str(best_model_path))
            self.assertEqual(result["optimization_curves_plot_path"], str(curves_path))
            self.assertTrue(best_model_path.exists())
            self.assertTrue(curves_path.exists())


if __name__ == "__main__":
    unittest.main()