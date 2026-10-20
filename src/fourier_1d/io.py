"""Minimal I/O helpers for two-column time-series files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def load_signal(path: str | Path) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load a whitespace-delimited file with columns ``(t, y)``."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected a two-column file with time and amplitude.")
    return data[:, 0], data[:, 1]


def save_signal(
    path: str | Path,
    time: ArrayLike,
    amplitude: ArrayLike,
) -> Path:
    """Persist a time series as a two-column text file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.column_stack([np.asarray(time, dtype=float), np.asarray(amplitude, dtype=float)])
    np.savetxt(out, stacked, fmt="%.8e")
    return out
