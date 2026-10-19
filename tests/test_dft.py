"""Numerical checks against NumPy's FFT reference implementation."""

from __future__ import annotations

import numpy as np
import pytest

from fourier_1d.dft import dft, fft, idft, ifft
from fourier_1d.signals import multi_tone


@pytest.fixture
def tone() -> np.ndarray:
    _, y = multi_tone((5.0, 12.0), (1.0, 0.5), sample_rate=64.0, duration=1.0, noise_std=0.0)
    return y


def test_dft_matches_numpy(tone: np.ndarray) -> None:
    np.testing.assert_allclose(dft(tone), np.fft.fft(tone), rtol=1e-10, atol=1e-10)


def test_fft_matches_numpy_power_of_two(tone: np.ndarray) -> None:
    assert tone.size & (tone.size - 1) == 0
    np.testing.assert_allclose(fft(tone), np.fft.fft(tone), rtol=1e-10, atol=1e-10)


def test_fft_falls_back_for_non_power_of_two() -> None:
    x = np.linspace(0.0, 1.0, 30)
    np.testing.assert_allclose(fft(x), np.fft.fft(x), rtol=1e-10, atol=1e-10)


def test_idft_roundtrip(tone: np.ndarray) -> None:
    recovered = idft(dft(tone))
    np.testing.assert_allclose(recovered.real, tone, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(recovered.imag, 0.0, atol=1e-10)


def test_ifft_roundtrip(tone: np.ndarray) -> None:
    recovered = ifft(fft(tone))
    np.testing.assert_allclose(recovered.real, tone, rtol=1e-10, atol=1e-10)


def test_rejects_empty_and_nd() -> None:
    with pytest.raises(ValueError):
        dft([])
    with pytest.raises(ValueError):
        dft(np.ones((4, 4)))
