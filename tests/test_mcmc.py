import numpy as np
import pytest

from analysis.mcmc import ess, hpd, read_log


def test_hpd_normal():
    x = np.random.default_rng(0).normal(size=200_000)
    lo, hi = hpd(x)
    assert lo == pytest.approx(-1.96, abs=0.03)
    assert hi == pytest.approx(1.96, abs=0.03)


def test_hpd_skewed_is_shorter_than_equal_tailed():
    x = np.random.default_rng(1).exponential(size=100_000)
    lo, hi = hpd(x)
    assert lo == pytest.approx(0.0, abs=0.01)
    assert hi == pytest.approx(-np.log(0.05), abs=0.05)
    assert hi - lo < np.quantile(x, 0.975) - np.quantile(x, 0.025)


def test_ess_iid_and_ar1():
    rng = np.random.default_rng(2)
    n = 20_000
    assert ess(rng.normal(size=n)) == pytest.approx(n, rel=0.1)
    phi = 0.9
    x = np.empty(n)
    x[0] = 0
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    expected = n * (1 - phi) / (1 + phi)  # AR(1) integrated autocorrelation time
    assert ess(x) == pytest.approx(expected, rel=0.25)


def test_read_log_burnin(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("# comment\nSample\ta\tb\n" + "".join(f"{i}\t{i}\t{2*i}\n" for i in range(10)))
    log = read_log(p, burnin_frac=0.2)
    assert log["a"].tolist() == list(range(2, 10))
    assert log["b"][0] == 4
