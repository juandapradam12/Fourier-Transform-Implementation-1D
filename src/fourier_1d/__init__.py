"""One-dimensional Discrete Fourier Transform toolkit.

Implements the DFT from first principles (matrix form) and a radix-2 FFT,
with spectrum utilities for signal-analysis workflows.
"""

from fourier_1d.dft import dft, fft, idft, ifft
from fourier_1d.spectrum import (
    frequency_axis,
    magnitude_spectrum,
    phase_spectrum,
    power_spectrum,
)
from fourier_1d.signals import multi_tone, sample_signal

__all__ = [
    "dft",
    "idft",
    "fft",
    "ifft",
    "frequency_axis",
    "magnitude_spectrum",
    "phase_spectrum",
    "power_spectrum",
    "multi_tone",
    "sample_signal",
]

__version__ = "1.0.0"
