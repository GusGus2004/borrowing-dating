"""MCMC output utilities: read BEAST2 logs, effective sample size, HPD intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_log(path: Path | str, burnin_frac: float = 0.1) -> dict[str, np.ndarray]:
    """Read a BEAST2 trace log (tab-separated, '#' comments) and drop burn-in."""
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#") and l.strip()]
    header = lines[0].rstrip("\n").split("\t")
    data = np.array([[float(x) for x in l.rstrip("\n").split("\t")] for l in lines[1:]])
    start = int(len(data) * burnin_frac)
    return {h: data[start:, i] for i, h in enumerate(header)}


def ess(x: np.ndarray) -> float:
    """Effective sample size via Geyer's initial positive sequence estimator."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float(n)
    xc = x - x.mean()
    f = np.fft.rfft(xc, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1) * x.var())
    # sum consecutive pairs; stop at first non-positive pair
    tau = -1.0
    for k in range(0, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return float(n / max(tau, 1.0))


def hpd(x: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Shortest interval containing ``mass`` of the samples."""
    s = np.sort(np.asarray(x, float))
    n = len(s)
    k = int(np.ceil(mass * n))
    if k >= n:
        return float(s[0]), float(s[-1])
    widths = s[k - 1:] - s[: n - k + 1]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k - 1])
