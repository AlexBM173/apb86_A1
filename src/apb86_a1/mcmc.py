from __future__ import annotations

"""MCMC helper functions for emulator-based posterior sampling.

This module provides utilities to:
- reconstruct full power spectra from emulator outputs via PCA inverse transform
- build adaptive uniform/log-uniform priors from training parameter ranges
- sample valid initial walker positions from the chosen priors
- evaluate log-prior, log-likelihood, and log-posterior for emcee
"""

from typing import Any, cast

import numpy as np
import torch


def reconstruct_power_spectrum(
    sampled_params: np.ndarray,
    model: Any,
    pca_model: Any,
    normalisation_stats: Any,
) -> np.ndarray:
    """Map sampled parameters to a reconstructed full power spectrum."""

    emulator_input = torch.tensor(sampled_params[:4], dtype=torch.float32).reshape(1, -1)
    model_device = next(model.parameters()).device
    emulator_input = emulator_input.to(model_device)

    model.eval()
    with torch.no_grad():
        predicted_pca = model(emulator_input).cpu().numpy()

    reconstructed_normalised = pca_model.inverse_transform(predicted_pca).flatten()
    reconstructed_power = (
        reconstructed_normalised * normalisation_stats.scale + normalisation_stats.min_power
    )
    return reconstructed_power


def build_parameter_priors(training_params: np.ndarray, order_threshold: float = 2.0) -> list[dict[str, float | int | str]]:
    """Create adaptive priors from training parameter ranges.

    If a positive-valued parameter spans at least ``order_threshold`` orders of
    magnitude, choose a log-uniform prior; otherwise choose uniform.
    """

    training_params = np.asarray(training_params, dtype=float)
    mins = training_params.min(axis=0)
    maxs = training_params.max(axis=0)

    prior_specs: list[dict[str, float | int | str]] = []
    for idx, (p_min, p_max) in enumerate(zip(mins, maxs)):
        span_orders = np.inf
        prior_type = "uniform"

        if p_min > 0 and p_max > 0:
            span_orders = np.log10(p_max / p_min)
            if span_orders >= order_threshold:
                prior_type = "log_uniform"

        prior_specs.append(
            {
                "index": idx,
                "min": float(p_min),
                "max": float(p_max),
                "span_orders": float(span_orders),
                "prior": prior_type,
            }
        )

    return prior_specs


def noise_prior_type(noise_bounds: tuple[float, float], order_threshold: float = 2.0) -> str:
    """Select uniform or log-uniform noise prior from bounds."""

    noise_min, noise_max = noise_bounds
    if noise_min > 0 and noise_max > 0:
        span_orders = np.log10(noise_max / noise_min)
        if span_orders >= order_threshold:
            return "log_uniform"
    return "uniform"


def sample_initial_positions(
    n_walkers: int,
    prior_specs: list[dict[str, float | int | str]],
    noise_bounds: tuple[float, float],
    order_threshold: float = 2.0,
) -> np.ndarray:
    """Draw initial walker positions from the selected priors."""

    positions = np.zeros((n_walkers, len(prior_specs) + 1), dtype=float)
    noise_min, noise_max = noise_bounds
    noise_prior = noise_prior_type(noise_bounds, order_threshold=order_threshold)

    for walker_idx in range(n_walkers):
        for spec in prior_specs:
            index = int(spec["index"])
            p_min = float(spec["min"])
            p_max = float(spec["max"])
            prior_name = str(spec["prior"])

            if prior_name == "log_uniform":
                log_sample = cast(float, np.random.uniform(np.log(p_min), np.log(p_max)))
                positions[walker_idx, index] = float(np.exp(log_sample))
            else:
                positions[walker_idx, index] = np.random.uniform(p_min, p_max)

        if noise_prior == "log_uniform":
            noise_log_sample = cast(float, np.random.uniform(np.log(noise_min), np.log(noise_max)))
            positions[walker_idx, -1] = float(np.exp(noise_log_sample))
        else:
            positions[walker_idx, -1] = np.random.uniform(noise_min, noise_max)

    return positions


def log_prior(
    sampled_params: np.ndarray,
    prior_specs: list[dict[str, float | int | str]],
    noise_bounds: tuple[float, float],
    order_threshold: float = 2.0,
) -> float:
    """Evaluate adaptive per-parameter priors and adaptive noise prior."""

    logp = 0.0

    for spec in prior_specs:
        index = int(spec["index"])
        value = float(sampled_params[index])
        p_min = float(spec["min"])
        p_max = float(spec["max"])
        prior_name = str(spec["prior"])

        if value < p_min or value > p_max:
            return float(-np.inf)

        if prior_name == "log_uniform":
            if value <= 0:
                return float(-np.inf)
            logp += -np.log(value) - np.log(np.log(p_max / p_min))
        else:
            logp += -np.log(p_max - p_min)

    noise_value = float(sampled_params[4])
    noise_min, noise_max = noise_bounds
    if noise_value < noise_min or noise_value > noise_max:
        return float(-np.inf)

    noise_prior = noise_prior_type(noise_bounds, order_threshold=order_threshold)
    if noise_prior == "log_uniform":
        if noise_value <= 0:
            return float(-np.inf)
        logp += -np.log(noise_value) - np.log(np.log(noise_max / noise_min))
    else:
        logp += -np.log(noise_max - noise_min)

    return float(logp)


def log_likelihood(
    sampled_params: np.ndarray,
    model: Any,
    pca_model: Any,
    observed_power_spectrum: np.ndarray,
    normalisation_stats: Any,
) -> float:
    """Evaluate Gaussian log-likelihood in full power-spectrum space."""

    n_k = 100
    f_noise = float(sampled_params[4])

    if f_noise < 0:
        return float(-np.inf)

    predicted_power = reconstruct_power_spectrum(
        sampled_params,
        model,
        pca_model,
        normalisation_stats,
    )

    noise_power = f_noise * predicted_power
    variance = 2.0 * (predicted_power + noise_power) ** 2 / n_k
    variance = np.maximum(variance, 1e-12)

    residual = np.asarray(observed_power_spectrum, dtype=float) - predicted_power
    return float(-0.5 * np.sum(residual**2 / variance + np.log(2.0 * np.pi * variance)))


def log_posterior(
    sampled_params: np.ndarray,
    model: Any,
    pca_model: Any,
    observed_power_spectrum: np.ndarray,
    normalisation_stats: Any,
    prior_specs: list[dict[str, float | int | str]],
    noise_bounds: tuple[float, float],
) -> float:
    """Evaluate posterior as log-prior + log-likelihood."""

    prior_value = log_prior(sampled_params, prior_specs, noise_bounds)
    if not np.isfinite(prior_value):
        return float(-np.inf)

    likelihood_value = log_likelihood(
        sampled_params,
        model,
        pca_model,
        observed_power_spectrum,
        normalisation_stats,
    )
    return float(prior_value + likelihood_value)


__all__ = [
    "build_parameter_priors",
    "log_likelihood",
    "log_posterior",
    "log_prior",
    "noise_prior_type",
    "reconstruct_power_spectrum",
    "sample_initial_positions",
]
