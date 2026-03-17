from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from apb86_a1.io import (
    load_observations,
    load_simulation_dataset,
    load_split_dataset,
    save_observations_pca,
    save_pca_model,
    save_split_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class IOTests(unittest.TestCase):
    def test_load_observations_returns_expected_arrays(self) -> None:
        observations = load_observations(DATA_DIR)

        self.assertEqual(observations.k.ndim, 1)
        self.assertEqual(observations.power.ndim, 1)
        self.assertGreater(observations.k.size, 0)
        self.assertEqual(observations.k.shape, observations.power.shape)

    def test_load_simulation_dataset_flattens_parameters(self) -> None:
        simulations = load_simulation_dataset(DATA_DIR)

        self.assertEqual(simulations.params.ndim, 2)
        self.assertEqual(simulations.params.shape[1], 4)
        self.assertEqual(simulations.spectra.shape[0], simulations.params.shape[0])
        self.assertEqual(simulations.ks.shape, simulations.spectra.shape)
        self.assertEqual(len(simulations.filenames), simulations.params.shape[0])

    def test_save_and_load_split_datasets_round_trip(self) -> None:
        x_train = np.arange(8, dtype=float).reshape(2, 4)
        y_train = np.arange(4, dtype=float).reshape(2, 2)
        x_val = np.arange(4, dtype=float).reshape(1, 4)
        y_val = np.arange(2, dtype=float).reshape(1, 2)
        x_test = np.arange(4, 8, dtype=float).reshape(1, 4)
        y_test = np.arange(2, 4, dtype=float).reshape(1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_split_datasets(
                tmpdir,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
            )

            train = load_split_dataset(Path(tmpdir) / "train.npz")
            val = load_split_dataset(Path(tmpdir) / "val.npz")
            test = load_split_dataset(Path(tmpdir) / "test.npz")

            np.testing.assert_array_equal(train["X_train"], x_train)
            np.testing.assert_array_equal(train["y_train"], y_train)
            np.testing.assert_array_equal(val["X_val"], x_val)
            np.testing.assert_array_equal(val["y_val"], y_val)
            np.testing.assert_array_equal(test["X_test"], x_test)
            np.testing.assert_array_equal(test["y_test"], y_test)

    def test_save_observations_pca_and_pca_model_create_files(self) -> None:
        observations_pca = np.array([[1.0, 2.0]])
        pca_model = PCA(n_components=1).fit(np.array([[0.0, 1.0], [1.0, 0.0]]))

        with tempfile.TemporaryDirectory() as tmpdir:
            observations_path = Path(tmpdir) / "observations_pca.npz"
            model_path = Path(tmpdir) / "pca_model.pkl"

            save_observations_pca(observations_path, observations_pca)
            save_pca_model(model_path, pca_model)

            self.assertTrue(observations_path.exists())
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()