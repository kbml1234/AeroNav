import pytest

from src.drone_env import DroneNavEnv
from src.evaluate import load_splits, route_efficiency, split_seeds, summarise


def test_splits_never_overlap():
    splits = load_splits()
    ranges = sorted((splits[name]["low"], splits[name]["high"]) for name in ["train", "validation", "test"])
    for (low1, high1), (low2, high2) in zip(ranges, ranges[1:]):
        assert high1 <= low2


def test_earlier_seeds_are_in_no_split():
    used = set(range(100)) | {6, 37, 42} | set(range(10_000, 10_300))
    splits = load_splits()
    for name in ["train", "validation", "test"]:
        assert not any(splits[name]["low"] <= s < splits[name]["high"] for s in used)


def test_validation_and_test_have_300_maps_each():
    assert split_seeds("validation") == list(range(20_000, 20_300))
    assert len(split_seeds("test")) == 300


def test_training_env_only_draws_training_seeds():
    train = load_splits()["train"]
    env = DroneNavEnv(seed_range=(train["low"], train["high"]))
    env.reset(seed=0)
    for _ in range(200):
        obs, info = env.reset()
        assert train["low"] <= info["map_seed"] < train["high"]


def test_summarise_counts_endings():
    results = [{"ending": "goal", "steps": 14}, {"ending": "goal", "steps": 16},
               {"ending": "collision", "steps": 3}, {"ending": "timeout", "steps": 200}]
    assert summarise(results) == {"n": 4, "success_rate": 0.5, "collisions": 1,
                                  "timeouts": 1, "mean_goal_steps": 15.0}


def test_a_perfect_flight_has_efficiency_1():
    route = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]   # 15 squares = 14 moves
    assert route_efficiency({"ending": "goal", "steps": 14}, route) == 1.0


def test_extra_moves_lower_the_efficiency():
    route = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]
    assert route_efficiency({"ending": "goal", "steps": 18}, route) == pytest.approx(14 / 18)


def test_a_crashed_or_timed_out_flight_has_no_efficiency():
    route = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 8)]
    assert route_efficiency({"ending": "collision", "steps": 2}, route) is None
    assert route_efficiency({"ending": "timeout", "steps": 200}, route) is None
