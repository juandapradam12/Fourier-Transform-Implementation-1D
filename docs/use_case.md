# Use case: transparent 1-D spectral analysis

## Context

The original artifact was a short Spanish notebook-style script that loaded an external `funcion.dat` file and applied a hand-written DFT matrix. That fixture was never committed, so the repository could not reproduce a spectrum on a clean clone.

This rewrite keeps the **same scientific intent**—compute $X_k$ from first principles—while turning the idea into a small, installable package with tests, a synthetic signal, and documentation suitable for a public portfolio.

## Problem statement

For a real-valued sequence $x[n]$ sampled at rate $f_s$, estimate the discrete spectrum $X[k]$ and present it on a frequency axis in Hertz, with:

- an implementation that does **not** call `numpy.fft` for the primary transform,
- numerical agreement with `numpy.fft` as an external oracle,
- utilities for magnitude, phase, and power spectra used in routine signal inspection.

## Solution outline

| Layer | Responsibility |
| --- | --- |
| `dft` | Matrix DFT / IDFT and radix-2 FFT / IFFT |
| `spectrum` | Map bins to Hz; magnitude, phase, power |
| `signals` | Controlled multi-tone + noise fixtures |
| `io` | Optional persistence of $(t, y)$ pairs |

Forward transform (matrix form, complexity $O(N^2)$):

$$
X = W x, \qquad W_{kn} = \exp\!\left(-2\pi i \frac{kn}{N}\right).
$$

The iterative radix-2 FFT (complexity $O(N \log N)$ when $N$ is a power of two) bit-reverses the input, then applies butterfly stages with twiddle factors

$$
\omega_n = e^{-2\pi i n / N}.
$$

Non-power-of-two lengths fall back to the matrix DFT so the public `fft` API remains total. A small benchmark (`examples/run_benchmark.py`) contrasts wall time of DFT, iterative FFT, and `numpy.fft` to make the complexity gap observable.

## Signal model used in the demo

The default fixture is

$$
y(t) = \sin(2\pi \cdot 5\, t) + 0.55\sin(2\pi \cdot 12\, t) + \varepsilon(t),
$$

with $\varepsilon(t) \sim \mathcal{N}(0, 0.08^2)$, $f_s = 256\,\mathrm{Hz}$, and duration $1\,\mathrm{s}$. Expected magnitude peaks sit near $5\,\mathrm{Hz}$ and $12\,\mathrm{Hz}$.

## Acceptance criteria (encoded as tests)

1. $\| \mathrm{DFT}(x) - \mathtt{numpy.fft.fft}(x) \|_\infty$ is negligible for representative $x$.
2. Same for the radix-2 FFT on power-of-two lengths.
3. $\mathrm{IDFT}(\mathrm{DFT}(x)) \approx x$ (round trip).
4. Parseval: $\sum_n |x_n|^2 = N^{-1}\sum_k |X_k|^2$.
5. Magnitude peaks recover the injected tones within bin resolution.

## Why this is a portfolio piece

It shows end-to-end ownership of a classical numerical method: correct equations, deliberate software boundaries, automated oracle checks, and a documented use case rather than a disposable script.
