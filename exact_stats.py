"""Small statistical helpers requiring only Python's stdlib and NumPy.

``binom_sf`` is a one-sided upper-tail test against a *specified*, constant
Bernoulli probability. Its numerical accuracy does not establish that the
independence or constant-probability assumptions hold for a particular rule.
Run ``python exact_stats.py`` for independent, deterministic self-tests.
"""

from __future__ import annotations

from functools import lru_cache
import itertools
import math
import operator
from statistics import NormalDist

import numpy as np


def _integer(value: int, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _probability(value: float, name: str = "p") -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


@lru_cache(maxsize=2048)
def _log_combinations(n: int) -> np.ndarray:
    """Log binomial coefficients, computed with stdlib log-gamma."""
    common = math.lgamma(n + 1)
    values = np.fromiter(
        (common - math.lgamma(j + 1) - math.lgamma(n - j + 1)
         for j in range(n + 1)),
        dtype=np.float64,
        count=n + 1,
    )
    values.flags.writeable = False
    return values


def _logsumexp(values: np.ndarray) -> float:
    largest = float(np.max(values))
    if largest == -math.inf:
        return -math.inf
    return largest + math.log(float(np.exp(values - largest).sum()))


@lru_cache(maxsize=131072)
def _binom_sf_cached(k: int, n: int, p: float) -> float:
    # An empty statistical sample provides no evidence; the analysis API
    # deliberately reports its test p-value as 1 even if k > n.
    if n == 0 or k <= 0:
        return 1.0
    if k > n or p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    if k == 1:
        return -math.expm1(n * log_q)
    if k == n:
        return math.exp(n * log_p)
    log_comb = _log_combinations(n)
    if k <= n * p:
        # Here the lower tail is at most about one half. Using expm1 also
        # avoids losing precision if a boundary case makes it close to 1.
        j = np.arange(k, dtype=np.float64)
        log_cdf = _logsumexp(log_comb[:k] + j * log_p + (n - j) * log_q)
        answer = -math.expm1(min(0.0, log_cdf))
    else:
        # Compute a small upper tail directly rather than as 1 - CDF.
        j = np.arange(k, n + 1, dtype=np.float64)
        log_sf = _logsumexp(log_comb[k:] + j * log_p + (n - j) * log_q)
        answer = math.exp(min(0.0, log_sf))
    return min(1.0, max(0.0, answer))


def binom_sf(k: int, n: int, p: float) -> float:
    """Return P(Binomial(n, p) >= k), with empty-sample test p-value 1.

    ``k`` may lie outside [0, n]. ``n`` must be nonnegative. All terms
    in the chosen tail are summed, so this is not a normal approximation.
    Results below the float64 representable range underflow to zero.
    """
    k, n = _integer(k, "k"), _integer(n, "n")
    if n < 0:
        raise ValueError("n must be nonnegative")
    return _binom_sf_cached(k, n, _probability(p))


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    The interval is descriptive when the binomial sampling assumptions
    do not hold. An empty sample returns the uninformative interval (0, 1).
    """
    k, n = _integer(k, "k"), _integer(n, "n")
    level = float(level)
    if n < 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n")
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    ratio = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (ratio + z2 / (2.0 * n)) / denominator
    radius = z * math.sqrt(ratio * (1.0 - ratio) / n + z2 / (4.0 * n * n)) / denominator
    lower = 0.0 if k == 0 else max(0.0, center - radius)
    upper = 1.0 if k == n else min(1.0, center + radius)
    return lower, upper


def poisson_binomial_sf(k: int, probabilities) -> float:
    """P(sum of independent, nonidentical Bernoulli trials >= k), by DP.

    A variable probability vector alone does not account for dependent
    outcomes or adaptive selection of the trials. With zero trials this
    function returns the mathematical tail (1 if k <= 0, otherwise 0),
    unlike ``binom_sf``'s empty-sample *test* convention.
    """
    k = _integer(k, "k")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("probabilities must be finite and between 0 and 1")
    if k <= 0:
        return 1.0
    if k > probabilities.size:
        return 0.0
    k -= int(np.count_nonzero(probabilities == 1.0))
    probabilities = probabilities[(probabilities > 0.0) & (probabilities < 1.0)]
    if k <= 0:
        return 1.0
    if k > probabilities.size:
        return 0.0
    distribution = np.zeros(probabilities.size + 1, dtype=np.float64)
    distribution[0] = 1.0
    for count, p in enumerate(probabilities, start=1):
        distribution[1:count + 1] = (
            distribution[1:count + 1] * (1.0 - p)
            + distribution[:count] * p
        )
        distribution[0] *= 1.0 - p
    return min(1.0, max(0.0, math.fsum(distribution[k:])))


def selftest() -> dict[str, int | str]:
    """Check against independent finite sums and enumerations."""
    count = 0
    for n in range(1, 26):
        for p in (0.0, 0.01, 0.1, 0.3, 0.5, 0.9, 0.99, 1.0):
            for k in range(-1, n + 2):
                expected = math.fsum(
                    math.comb(n, j) * p ** j * (1.0 - p) ** (n - j)
                    for j in range(max(0, k), n + 1)
                ) if k <= n else 0.0
                actual = binom_sf(k, n, p)
                assert math.isclose(actual, expected, rel_tol=3e-13, abs_tol=3e-15), (
                    k, n, p, actual, expected
                )
                count += 1
    assert binom_sf(0, 0, 0.5) == binom_sf(1, 0, 0.5) == 1.0
    assert math.isclose(binom_sf(3, 15, 0.1), 0.18406106910639114, rel_tol=2e-14)
    assert math.isclose(binom_sf(1, 1248, 1e-12), -math.expm1(1248 * math.log1p(-1e-12)), rel_tol=1e-14)
    assert 0.0 < binom_sf(800, 1248, 0.5) < 1e-20
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert math.isclose(lo, 0.4038315303659956, rel_tol=1e-13)
    assert math.isclose(hi, 0.5961684696340044, rel_tol=1e-13)
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0
    probabilities = [0.0, 0.05, 0.2, 0.55, 0.9, 1.0]
    for k in range(-1, len(probabilities) + 2):
        expected = math.fsum(
            math.prod(p if bit else 1.0 - p for bit, p in zip(bits, probabilities))
            for bits in itertools.product((0, 1), repeat=len(probabilities))
            if sum(bits) >= k
        )
        assert math.isclose(poisson_binomial_sf(k, probabilities), expected, rel_tol=2e-14, abs_tol=2e-15)
    for k in range(17):
        assert math.isclose(poisson_binomial_sf(k, [0.1] * 15), binom_sf(k, 15, 0.1), rel_tol=3e-13, abs_tol=2e-15)
    assert poisson_binomial_sf(0, []) == 1.0
    assert poisson_binomial_sf(1, []) == 0.0
    for function, args in (
        (binom_sf, (1, -1, 0.5)),
        (binom_sf, (1, 3, math.nan)),
        (wilson_interval, (4, 3)),
        (poisson_binomial_sf, (1, [-0.1])),
    ):
        try:
            function(*args)
        except ValueError:
            pass
        else:
            raise AssertionError((function.__name__, args))
    return {"status": "passed", "independent_binomial_cases": count}


if __name__ == "__main__":
    print(selftest())
