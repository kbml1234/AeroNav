from collections import Counter

import numpy as np

from src.astar import astar
from src.drone_env import DroneNavEnv
from src.q_learning import best_action, choose_action, epsilon_at, make_q_table, q_update, run_greedy, train


def test_new_q_table_is_all_zeros():
    q = make_q_table(8)
    assert q.shape == (8, 8, 4)    # 8 rows x 8 cols x 4 actions = 256 scores
    assert np.all(q == 0)


def test_best_action_picks_the_highest_score():
    q = make_q_table(8)
    q[2, 3] = [-30, 12, 9, -3]
    assert best_action(q, (2, 3), np.random.default_rng(0)) == 1


def test_best_action_only_looks_at_its_own_square():
    q = make_q_table(8)
    q[2, 3] = [-30, 12, 9, -3]
    q[5, 5] = [0, 0, 7, 0]
    assert best_action(q, (5, 5), np.random.default_rng(0)) == 2


def test_all_zeros_tie_is_broken_fairly():
    q = make_q_table(8)
    rng = np.random.default_rng(0)
    counts = Counter(best_action(q, (0, 0), rng) for _ in range(400))
    assert set(counts) == {0, 1, 2, 3}
    assert all(70 <= counts[a] <= 130 for a in range(4))


def test_tie_only_between_the_top_scores():
    q = make_q_table(8)
    q[0, 0] = [5, 5, 0, -30]
    rng = np.random.default_rng(0)
    picks = {best_action(q, (0, 0), rng) for _ in range(100)}
    assert picks == {0, 1}


def test_same_seed_same_choices():
    q = make_q_table(8)
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    picks1 = [best_action(q, (0, 0), rng1) for _ in range(20)]
    picks2 = [best_action(q, (0, 0), rng2) for _ in range(20)]
    assert picks1 == picks2
    assert len(set(picks1)) > 1


def test_epsilon_zero_always_exploits():
    q = make_q_table(8)
    q[2, 3] = [-30, 12, 9, -3]
    rng = np.random.default_rng(0)
    assert all(choose_action(q, (2, 3), 0.0, rng) == 1 for _ in range(200))


def test_epsilon_one_always_explores():
    q = make_q_table(8)
    q[2, 3] = [-30, 12, 9, -3]
    rng = np.random.default_rng(0)
    counts = Counter(choose_action(q, (2, 3), 1.0, rng) for _ in range(400))
    assert all(70 <= counts[a] <= 130 for a in range(4))


def test_epsilon_sets_how_often_it_explores():
    # epsilon = 0.2: east is best, so non-east picks come only from exploring.
    # exploring picks each of the 4 actions, so 3/4 of explores are non-east: 0.2 * 3/4 = 15%
    q = make_q_table(8)
    q[2, 3] = [-30, 12, 9, -3]
    rng = np.random.default_rng(0)
    picks = [choose_action(q, (2, 3), 0.2, rng) for _ in range(4000)]
    not_east = sum(a != 1 for a in picks) / len(picks)
    assert 0.12 <= not_east <= 0.18


def test_update_goal_move():
    # (6,7) south onto the goal, old score 50, alpha 0.5: target 100, new 75
    q = make_q_table(8)
    q[6, 7, 2] = 50
    new = q_update(q, (6, 7), 2, 100, (7, 7), True, alpha=0.5, gamma=0.9)
    assert new == 75
    assert q[6, 7, 2] == 75


def test_update_spreads_the_reward_backwards():
    # (5,7) south to (6,7): target = -1 + 0.9 * 50 = 44, halfway from 0 is 22
    q = make_q_table(8)
    q[6, 7, 2] = 50
    new = q_update(q, (5, 7), 2, -1, (6, 7), False, alpha=0.5, gamma=0.9)
    assert abs(new - 22) < 1e-9


def test_crash_has_no_future():
    # a crash leaves the drone on its old square; that square's good scores must not soften the crash
    q = make_q_table(8)
    q[2, 3] = [0, 50, 0, 0]
    new = q_update(q, (2, 3), 0, -30, (2, 3), True, alpha=0.5, gamma=0.9)
    assert new == -15              # target is just -30; halfway from 0


def test_timeout_keeps_its_future():
    q = make_q_table(8)
    q[6, 7, 2] = 50
    new = q_update(q, (5, 7), 2, -1, (6, 7), False, alpha=1.0, gamma=0.9)
    assert abs(new - 44) < 1e-9    # -1 + 0.9 * 50, alpha 1 jumps all the way


def test_update_changes_only_one_score():
    q = make_q_table(8)
    q_update(q, (5, 7), 2, -1, (6, 7), False, alpha=0.5, gamma=0.9)
    assert np.count_nonzero(q) == 1


def test_epsilon_schedule():
    assert epsilon_at(0, 1000, 1.0, 0.05) == 1.0
    assert abs(epsilon_at(400, 1000, 1.0, 0.05) - 0.525) < 1e-9
    assert epsilon_at(800, 1000, 1.0, 0.05) == 0.05
    assert epsilon_at(999, 1000, 1.0, 0.05) == 0.05


def test_training_learns_the_shortest_route_on_a_fixed_map():
    env = DroneNavEnv(size=8, obstacle_density=0.15)
    q, history = train(env, 42, 2000, alpha=0.1, gamma=0.95, seed=0)
    assert len(history) == 2000
    final = run_greedy(env, q, 42)
    shortest = len(astar(env.grid, env.start, env.goal)) - 1
    assert final["ending"] == "goal"
    assert final["steps"] == shortest


def test_training_is_reproducible():
    env = DroneNavEnv(size=8, obstacle_density=0.15)
    q1, h1 = train(env, 42, 300, alpha=0.1, gamma=0.95, seed=3)
    q2, h2 = train(env, 42, 300, alpha=0.1, gamma=0.95, seed=3)
    assert np.array_equal(q1, q2) and h1 == h2
