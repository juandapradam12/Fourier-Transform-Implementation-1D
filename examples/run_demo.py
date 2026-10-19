#!/usr/bin/env python3
"""Run the portfolio demo: time series, magnitude spectrum, NumPy cross-check."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourier_1d.dft import dft, fft  # noqa: E402
from fourier_1d.signals import sample_signal  # noqa: E402
from fourier_1d.spectrum import magnitude_spectrum  # noqa: E402


def main() -> Path:
    t, y, sample_rate = sample_signal(sample_rate=256.0, duration=1.0)
    freqs, mag = magnitude_spectrum(y, sample_rate=sample_rate, centered=True, transform=fft)

    own = fft(y)
    reference = np.fft.fft(y)
    max_abs_err = float(np.max(np.abs(own - reference)))

    out_dir = ROOT / "assets" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spectrum_demo.png"

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(t, y, color="#1f4e79", lw=1.2)
    axes[0].set_title("Synthetic multi-tone signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(freqs, mag, color="#b85c38", lw=1.2)
    axes[1].set_title(f"Magnitude spectrum (max |FFT − NumPy| = {max_abs_err:.2e})")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("|X(f)|")
    axes[1].set_xlim(-sample_rate / 2, sample_rate / 2)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # Keep the matrix DFT path exercised for documentation traces.
    _ = dft(y[:64])

    print(f"Wrote {out_path}")
    print(f"Max absolute error vs numpy.fft.fft: {max_abs_err:.3e}")
    return out_path


if __name__ == "__main__":
    main()
