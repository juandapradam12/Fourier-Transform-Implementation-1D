"""Spectrum utility checks."""

from __future__ import annotations

import numpy as np

from fourier_1d.signals import multi_tone
from fourier_1d.spectrum import frequency_axis, magnitude_spectrum, power_spectrum


def test_frequency_axis_nyquist() -> None:
    freqs = frequency_axis(8, sample_rate=8.0)
    assert freqs[0] == 0.0
    assert freqs[4] == -4.0


def test_magnitude_peaks_at_known_tones() -> None:
    sample_rate = 128.0
    _, y = multi_tone((10.0, 25.0), (1.0, 0.7), sample_rate=sample_rate, duration=1.0)
    freqs, mag = magnitude_spectrum(y, sample_rate=sample_rate, centered=True)
    positive = freqs > 0
    peak_freqs = freqs[positive][np.argsort(mag[positive])[-2:]]
    assert set(np.round(peak_freqs).astype(int).tolist()) == {10, 25}


def test_parseval_relation() -> None:
    _, y = multi_tone((8.0,), sample_rate=64.0, duration=1.0)
    _, power = power_spectrum(y, sample_rate=64.0, centered=False, density=False)
    time_energy = np.sum(np.abs(y) ** 2)
    freq_energy = np.sum(power) / y.size
    np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-10)
