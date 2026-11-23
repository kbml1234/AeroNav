from collections import Counter

from src.drone_env import DroneNavEnv
from src.render import record_episode, save_gif, save_png


def test_record_crash():
    run = record_episode(DroneNavEnv(), [1, 1, 2, 3], seed=42)
    assert run["ending"] == "collision"
    assert run["crash_action"] == 3
    assert run["path"] == [(0, 0), (0, 1), (0, 2), (1, 2)]


def test_record_goal():
    run = record_episode(DroneNavEnv(), [1] * 7 + [2] * 7, seed=42)
    assert run["ending"] == "goal"
    assert run["path"][-1] == (7, 7)
    assert len(run["path"]) == 15


def test_record_timeout_and_visit_counts():
    run = record_episode(DroneNavEnv(max_steps=4), [1, 3] * 10, seed=42)
    assert run["ending"] == "timeout"
    counts = Counter(run["path"])
    assert counts[(0, 0)] == 3
    assert counts[(0, 1)] == 2


def test_save_png_and_gif(tmp_path):
    run = record_episode(DroneNavEnv(), [1, 1, 2, 3], seed=42)
    save_png(run, tmp_path / "crash.png")
    save_gif(run, tmp_path / "crash.gif")
    assert (tmp_path / "crash.png").stat().st_size > 0
    assert (tmp_path / "crash.gif").stat().st_size > 0


def test_record_crash_on_first_move():
    # crashing straight away still leaves the start square in the path
    run = record_episode(DroneNavEnv(), [0], seed=42)
    assert run["path"] == [(0, 0)]
    assert run["ending"] == "collision"
    assert run["crash_action"] == 0
