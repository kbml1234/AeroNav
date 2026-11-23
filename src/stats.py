
from math import comb, sqrt


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        raise ValueError("n must be at least 1")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    low = 0.0 if successes == 0 else centre - half        # exact ends (no rounding fuzz) at 0% and 100%
    high = 1.0 if successes == n else centre + half
    return low, high


def paired_counts(a_success, b_success):
    # counts maps where only A or only B succeeded
    if len(a_success) != len(b_success):
        raise ValueError("the two lists must cover the same flights")
    only_a = sum(1 for a, b in zip(a_success, b_success) if a and not b)
    only_b = sum(1 for a, b in zip(a_success, b_success) if b and not a)
    return only_a, only_b


def paired_p_value(only_a, only_b):
    # exact McNemar test, two-sided
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def seed_spread(values):
    return sum(values) / len(values), min(values), max(values)
