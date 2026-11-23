from src.learning_curves import plot_learning_curves, rolling_mean


def test_rolling_mean_slides_one_round_at_a_time():
    assert rolling_mean([0, 0, 1, 1], 2) == [0, 0, 0.5, 1]


def test_rolling_mean_uses_fewer_values_at_the_start():
    assert rolling_mean([4, 2, 6], 50) == [4, 3, 4]


def test_rolling_mean_skips_missing_values():
    # steps only exist on goal rounds; crash rounds are None and must not count as 0
    assert rolling_mean([None, 10, None, 20], 3) == [None, 10, 10, 15]


def test_plot_saves_a_file(tmp_path):
    history = [{"episode": i, "reward": -30 if i < 5 else 87, "steps": 1 if i < 5 else 14,
                "ending": "collision" if i < 5 else "goal", "epsilon": 1.0} for i in range(10)]
    out = tmp_path / "curves.png"
    plot_learning_curves(history, 14, out, window=3)
    assert out.stat().st_size > 0
