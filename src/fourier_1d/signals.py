"""Synthetic signals for demos and numerical checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def multi_tone(
    frequencies: tuple[float, ...] | list[float],
    amplitudes: tuple[float, ...] | list[float] | None = None,
    *,
    duration: float = 1.0,
    sample_rate: float = 256.0,
    phases: tuple[float, ...] | list[float] | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build a sum of sinusoids, optionally with additive Gaussian noise.

    Returns
    -------
    t, y:
        Time axis (seconds) and real-valued samples.
    """
    freqs = list(frequencies)
    if not freqs:
        raise ValueError("At least one frequency is required.")

    n_tones = len(freqs)
    amps = list(amplitudes) if amplitudes is not None else [1.0] * n_tones
    phis = list(phases) if phases is not None else [0.0] * n_tones

    if len(amps) != n_tones or len(phis) != n_tones:
        raise ValueError("amplitudes and phases must match frequencies length.")
    if sample_rate <= 0 or duration <= 0:
        raise ValueError("sample_rate and duration must be positive.")

    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=float) / sample_rate
    y = np.zeros(n, dtype=float)
    for f, a, phi in zip(freqs, amps, phis):
        y += a * np.sin(2.0 * np.pi * f * t + phi)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise_std, size=n)

    return t, y


def sample_signal(
    *,
    sample_rate: float = 256.0,
    duration: float = 1.0,
    seed: int = 7,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Canonical demo signal: 5 Hz + 12 Hz tones with light noise.

    Replaces the legacy ``funcion.dat`` fixture that was never versioned
    in the original repository.
    """
    t, y = multi_tone(
        (5.0, 12.0),
        (1.0, 0.55),
        duration=duration,
        sample_rate=sample_rate,
        noise_std=0.08,
        seed=seed,
    )
    return t, y, sample_rate
