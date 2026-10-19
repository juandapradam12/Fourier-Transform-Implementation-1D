#!/usr/bin/env python3
"""Compare wall time of matrix DFT, iterative FFT, and NumPy FFT."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourier_1d.dft import dft, fft  # noqa: E402


def _median_runtime(fn, x: np.ndarray, repeats: int = 5) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(x)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> Path:
    # DFT is O(N^2); keep the sweep modest so the script stays interactive.
    sizes = [16, 32, 64, 128, 256, 512]
    dft_times: list[float] = []
    fft_times: list[float] = []
    numpy_times: list[float] = []

    rng = np.random.default_rng(0)
    for n in sizes:
        x = rng.normal(size=n)
        dft_times.append(_median_runtime(dft, x, repeats=3 if n >= 256 else 5))
        fft_times.append(_median_runtime(fft, x))
        numpy_times.append(_median_runtime(np.fft.fft, x))
        print(
            f"N={n:4d}  DFT={dft_times[-1]:.4e}s  "
            f"FFT={fft_times[-1]:.4e}s  NumPy={numpy_times[-1]:.4e}s"
        )

    out_dir = ROOT / "assets" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_runtime.png"
    docs_path = ROOT / "docs" / "figures" / "benchmark_runtime.png"
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.loglog(sizes, dft_times, "o-", color="#1f4e79", label="Matrix DFT  $O(N^2)$")
    ax.loglog(sizes, fft_times, "s-", color="#b85c38", label="Iterative FFT  $O(N\\log N)$")
    ax.loglog(sizes, numpy_times, "^-", color="#2f6b4f", label="NumPy FFT")
    ax.set_xlabel("Signal length $N$")
    ax.set_ylabel("Median runtime [s]")
    ax.set_title("Runtime comparison: DFT vs iterative FFT vs NumPy")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    fig.savefig(out_path, dpi=140)
    fig.savefig(docs_path, dpi=140)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Wrote {docs_path}")
    return docs_path


if __name__ == "__main__":
    main()
