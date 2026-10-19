"""Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT).

The matrix DFT mirrors the classic $O(N^2)$ definition; the radix-2 FFT
recovers the familiar $O(N \\log N)$ complexity for power-of-two lengths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_1d_complex(signal: ArrayLike) -> NDArray[np.complex128]:
    arr = np.asarray(signal, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D signal, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Signal must contain at least one sample.")
    return arr


def dft_matrix(n: int, *, inverse: bool = False) -> NDArray[np.complex128]:
    """Build the unitary DFT / IDFT matrix of size ``n``.

    Forward entries follow

    .. math::

        W_{kn} = e^{-2\\pi i kn / N}

    The inverse uses the conjugate and the conventional $1/N$ scaling is
    applied outside this helper (see :func:`idft`).
    """
    if n < 1:
        raise ValueError("Matrix size must be a positive integer.")

    indices = np.arange(n)
    k = indices.reshape(n, 1)
    sign = 1.0 if inverse else -1.0
    return np.exp(sign * 2j * np.pi * k * indices / n)


def dft(signal: ArrayLike) -> NDArray[np.complex128]:
    """Compute the 1-D Discrete Fourier Transform via matrix multiplication.

    This is the direct implementation of

    .. math::

        X_k = \\sum_{n=0}^{N-1} x_n e^{-2\\pi i kn / N},
        \\qquad k = 0, \\ldots, N-1.
    """
    x = _as_1d_complex(signal)
    return dft_matrix(x.size) @ x


def idft(spectrum: ArrayLike) -> NDArray[np.complex128]:
    """Compute the inverse 1-D DFT.

    .. math::

        x_n = \\frac{1}{N} \\sum_{k=0}^{N-1} X_k e^{2\\pi i kn / N}.
    """
    x = _as_1d_complex(spectrum)
    n = x.size
    return (dft_matrix(n, inverse=True) @ x) / n


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fft_recursive(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    n = x.size
    if n == 1:
        return x.copy()

    even = _fft_recursive(x[0::2])
    odd = _fft_recursive(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    return np.concatenate([even + twiddle * odd, even - twiddle * odd])


def fft(signal: ArrayLike) -> NDArray[np.complex128]:
    """Radix-2 Cooley–Tukey FFT for power-of-two lengths.

    Falls back to :func:`dft` when ``N`` is not a power of two, so callers
    always receive a correct spectrum.
    """
    x = _as_1d_complex(signal)
    if not _is_power_of_two(x.size):
        return dft(x)
    return _fft_recursive(x)


def ifft(spectrum: ArrayLike) -> NDArray[np.complex128]:
    """Inverse FFT via conjugation of the forward transform.

    .. math::

        x = \\frac{1}{N}\\,\\mathrm{conj}\\bigl(\\mathrm{FFT}(\\mathrm{conj}(X))\\bigr).
    """
    x = _as_1d_complex(spectrum)
    return np.conjugate(fft(np.conjugate(x))) / x.size
