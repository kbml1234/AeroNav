import pytest

from src.stats import paired_counts, paired_p_value, seed_spread, wilson_interval


def test_wilson_calm_dqn_example():
    # 276 of 300 (92.0%) gives about 88.4% to 94.6%
    low, high = wilson_interval(276, 300)
    assert round(100 * low, 1) == 88.4 and round(100 * high, 1) == 94.6


def test_wilson_stays_inside_0_and_100():
    assert wilson_interval(0, 300)[0] == 0.0
    assert wilson_interval(300, 300)[1] == 1.0
    low, high = wilson_interval(0, 300)
    assert 0.0 < high < 0.02        # 0 of 300 does not prove "never": up to about 1.3%


def test_wilson_is_wider_with_fewer_maps():
    small = wilson_interval(45, 50)
    big = wilson_interval(270, 300)
    assert (small[1] - small[0]) > (big[1] - big[0])


def test_wilson_needs_flights():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_paired_counts_ignores_maps_where_both_agree():
    a = [True, True, False, False, True]
    b = [True, False, False, True, False]
    assert paired_counts(a, b) == (2, 1)


def test_paired_counts_needs_the_same_flights():
    with pytest.raises(ValueError):
        paired_counts([True], [True, False])


def test_paired_p_value_even_split_is_not_evidence():
    assert paired_p_value(10, 10) == 1.0
    assert paired_p_value(0, 0) == 1.0


def test_paired_p_value_lopsided_split_is_evidence():
    # 10 vs 0: chance by luck = 2 * (1/2)^10, about 0.002
    assert paired_p_value(10, 0) == pytest.approx(2 / 1024)
    assert paired_p_value(0, 10) == pytest.approx(2 / 1024)
    assert paired_p_value(15, 5) < 0.05 < paired_p_value(12, 8)


def test_seed_spread():
    mean, low, high = seed_spread([92.0, 91.0, 92.7])
    assert round(mean, 1) == 91.9 and low == 91.0 and high == 92.7
