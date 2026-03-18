from __future__ import annotations

import unittest

import numpy as np

from lba.preprocessing import (
    cumulative_explained_variance,
    fit_pca_with_observation,
    normalize_observation,
    normalize_spectra,
    split_training_data,
)


class PreprocessingTests(unittest.TestCase):
    def test_normalize_spectra_returns_expected_stats(self) -> None:
        spectra = np.array([[1.0, 3.0], [5.0, 7.0]])

        normalized, stats = normalize_spectra(spectra)

        self.assertEqual(stats.min_power, 1.0)
        self.assertEqual(stats.max_power, 7.0)
        np.testing.assert_allclose(
            normalized,
            np.array([[0.0, 2.0 / 6.0], [4.0 / 6.0, 1.0]]),
        )

    def test_normalize_observation_uses_existing_stats(self) -> None:
        observation = np.array([2.0, 5.0, 8.0])
        _, stats = normalize_spectra(np.array([[0.0, 10.0], [5.0, 7.5]]))

        normalized = normalize_observation(observation, stats)

        np.testing.assert_allclose(normalized, np.array([0.2, 0.5, 0.8]))

    def test_fit_pca_with_observation_preserves_expected_shapes(self) -> None:
        simulation_spectra = np.array(
            [[0.0, 0.5, 1.0], [0.1, 0.4, 0.9], [0.2, 0.3, 0.8]],
            dtype=float,
        )
        observation_spectrum = np.array([0.15, 0.35, 0.85], dtype=float)

        pca_results = fit_pca_with_observation(
            simulation_spectra,
            observation_spectrum,
            n_components=2,
        )

        self.assertEqual(pca_results.simulation_components.shape, (3, 2))
        self.assertEqual(pca_results.observation_components.shape, (1, 2))
        self.assertEqual(cumulative_explained_variance(pca_results.pca_model).shape, (2,))

    def test_split_training_data_produces_expected_partition_sizes(self) -> None:
        params = np.arange(40, dtype=float).reshape(10, 4)
        targets = np.arange(20, dtype=float).reshape(10, 2)

        splits = split_training_data(
            params,
            targets,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            random_state=42,
        )

        self.assertEqual(splits.x_train.shape, (8, 4))
        self.assertEqual(splits.x_val.shape, (1, 4))
        self.assertEqual(splits.x_test.shape, (1, 4))
        self.assertEqual(splits.y_train.shape, (8, 2))
        self.assertEqual(splits.y_val.shape, (1, 2))
        self.assertEqual(splits.y_test.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()