"""Frequency-domain helpers built on top of the DFT / FFT primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fourier_1d.dft import fft


def frequency_axis(
    n: int,
    sample_rate: float = 1.0,
    *,
    centered: bool = False,
) -> NDArray[np.floating]:
    """Return frequency bins in Hz for an ``n``-point transform.

    Parameters
    ----------
    n:
        Number of samples / FFT bins.
    sample_rate:
        Sampling frequency $f_s$ in Hz.
    centered:
        If ``True``, bins are ordered as in ``numpy.fft.fftshift``
        (negative frequencies first).
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    return np.fft.fftshift(freqs) if centered else freqs


def magnitude_spectrum(
    signal: ArrayLike,
    *,
    sample_rate: float = 1.0,
    centered: bool = True,
    transform=fft,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Magnitude spectrum $|X(f)|$ paired with its frequency axis."""
    spectrum = transform(signal)
    freqs = frequency_axis(spectrum.size, sample_rate, centered=centered)
    magnitude = np.abs(spectrum)
    if centered:
        magnitude = np.fft.fftshift(magnitude)
    return freqs, magnitude


def phase_spectrum(
    signal: ArrayLike,
    *,
    sample_rate: float = 1.0,
    centered: bool = True,
    unwrap: bool = True,
    transform=fft,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Phase spectrum $\\arg X(f)$ in radians."""
    spectrum = transform(signal)
    freqs = frequency_axis(spectrum.size, sample_rate, centered=centered)
    phase = np.angle(spectrum)
    if unwrap:
        phase = np.unwrap(phase)
    if centered:
        phase = np.fft.fftshift(phase)
    return freqs, phase


def power_spectrum(
    signal: ArrayLike,
    *,
    sample_rate: float = 1.0,
    centered: bool = True,
    density: bool = False,
    transform=fft,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """One-sided-friendly power spectrum $|X|^2$ (or density $|X|^2 / (f_s N)$).

    The two-sided spectrum is returned; callers that need a one-sided view
    can fold positive frequencies as needed for their domain.
    """
    spectrum = transform(signal)
    n = spectrum.size
    freqs = frequency_axis(n, sample_rate, centered=centered)
    power = np.abs(spectrum) ** 2
    if density:
        power = power / (sample_rate * n)
    if centered:
        power = np.fft.fftshift(power)
    return freqs, power
