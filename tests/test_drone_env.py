from collections import Counter
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from src.drone_env import move, step_outcome, run_episode, DroneNavEnv
from src.map_generator import generate_map

small_grid = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
])
small_goal = (3, 3)
def test_move():
    assert move((2, 2), 0) == (1, 2)
    assert move((2, 2), 1) == (2,3)       # right: what happens?
def test_wall_crash():
    new_pos, reward, terminated, event = step_outcome(small_grid, (1, 3), 3, small_goal)
    assert reward == -30
    assert terminated == True
    assert event == "collision"
    assert new_pos == (1, 3)
def test_reach_goal():
    new_pos, reward, terminated, event = step_outcome(small_grid, (2, 3), 2, small_goal)
    assert reward == 100
    assert terminated == True
    assert event == "goal"
    assert new_pos == (3,3)
def test_off_edge_crash():
    new_pos, reward, terminated, event = step_outcome(small_grid, (0, 0), 0, small_goal)
    assert reward == -30
    assert terminated == True
    assert event == "collision"
    assert new_pos == (0,0)
def test_normal_move():
    new_pos, reward, terminated, event = step_outcome(small_grid, (0,0), 1, small_goal)
    assert reward == -1
    assert terminated == False
    assert event == "move"
    assert new_pos == (0,1)
def test_timeout():
    log = run_episode(small_grid, (0, 0), small_goal, [1, 3, 1, 3], max_steps=3)
    assert len(log) == 3
    assert log[-1]["truncated"] == True
    assert log[-1]["terminated"] == False
    assert log[-1]["event"] == "move"

def test_default_limit_is_not_hit_early():
    log = run_episode(small_grid, (0, 0), small_goal, [1, 3, 1, 3])
    assert len(log) == 4
    assert log[-1]["truncated"] == False



def test_observation_shape_and_type():
    env = DroneNavEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_same_seed_same_map():
    env1 = DroneNavEnv()
    env2 = DroneNavEnv()
    env1.reset(seed=42)
    env2.reset(seed=42)
    assert np.array_equal(env1.grid, env2.grid)


def test_map_seed_option_gives_that_exact_map():
    env = DroneNavEnv(size=8, obstacle_density=0.15)
    env.reset(options={"map_seed": 7})
    grid, start, goal = generate_map(8, 0.15, 7)
    assert np.array_equal(env.grid, grid)


def test_start_observation():
    env = DroneNavEnv(size=8)
    obs, info = env.reset(seed=42)
    # drone at (0,0), goal at (7,7): positions divided by 7
    assert list(obs[:4]) == [0.0, 0.0, 1.0, 1.0]
    assert obs[4] == 1.0
    assert obs[7] == 1.0


def test_flagged_direction_crashes():
    env = DroneNavEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == -30
    assert terminated == True
    assert truncated == False
    assert info["event"] == "collision"


def test_env_timeout():
    env = DroneNavEnv(max_steps=1)
    obs, info = env.reset(seed=42)
    open_action = 1 if obs[5] == 0 else 2
    obs, reward, terminated, truncated, info = env.step(open_action)
    assert reward == -1
    assert terminated == False
    assert truncated == True


def test_check_env():
    check_env(DroneNavEnv())



def test_reset_after_an_episode_starts_fresh():
    env = DroneNavEnv()
    env.reset(seed=42)
    for action in [1, 1, 2, 3]:
        env.step(action)
    assert env.pos == (1, 2) and env.steps == 4
    obs, info = env.reset()
    assert env.pos == (0, 0)
    assert env.steps == 0
    assert list(obs[:2]) == [0.0, 0.0]


def test_reset_without_seed_gives_new_map_each_round():
    env = DroneNavEnv()
    env.reset(seed=42)
    maps = {env.grid.tobytes()}
    for _ in range(9):
        env.reset()
        maps.add(env.grid.tobytes())
    assert len(maps) == 10


def test_info_map_seed_rebuilds_the_same_map():
    env = DroneNavEnv()
    obs, info = env.reset(seed=3)
    grid, start, goal = generate_map(8, 0.15, info["map_seed"])
    assert np.array_equal(env.grid, grid)


def test_same_seed_same_sequence_of_maps():
    env1, env2 = DroneNavEnv(), DroneNavEnv()
    env1.reset(seed=5)
    env2.reset(seed=5)
    for _ in range(5):
        env1.reset()
        env2.reset()
        assert np.array_equal(env1.grid, env2.grid)


def test_all_four_actions_move_the_right_way():
    assert move((2, 2), 0) == (1, 2)
    assert move((2, 2), 1) == (2, 3)
    assert move((2, 2), 2) == (3, 2)
    assert move((2, 2), 3) == (2, 1)


def test_every_edge_is_a_crash():
    for pos, action in [((0, 0), 0),    # top: row -1 would silently wrap
                        ((0, 0), 3),    # left: col -1 would silently wrap
                        ((3, 0), 2),    # bottom: row 4 would raise IndexError
                        ((0, 3), 1)]:   # right: col 4 would raise IndexError
        new_pos, reward, terminated, event = step_outcome(small_grid, pos, action, small_goal)
        assert event == "collision"
        assert reward == -30
        assert new_pos == pos


def test_goal_on_last_step_is_success_not_timeout():
    env = DroneNavEnv(max_steps=14)
    env.reset(seed=42)
    for action in [1] * 7 + [2] * 7:
        obs, reward, terminated, truncated, info = env.step(action)
    assert info["event"] == "goal"
    assert terminated == True
    assert truncated == False


def test_observation_after_one_move_east():
    env = DroneNavEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs[0] == 0.0
    assert abs(obs[1] - 1 / 7) < 1e-6
    assert list(obs[4:]) == [1.0, 0.0, 1.0, 0.0]


def test_wall_flags_match_the_map_on_every_step():
    for seed in range(20):
        env = DroneNavEnv()
        obs, info = env.reset(seed=seed)
        for step in range(30):
            r, c = env.pos
            for flag, (dr, dc) in zip(obs[4:], [(-1, 0), (0, 1), (1, 0), (0, -1)]):
                nr, nc = r + dr, c + dc
                blocked = not (0 <= nr < 8 and 0 <= nc < 8) or env.grid[nr, nc] == 1
                assert flag == (1.0 if blocked else 0.0)
            open_actions = [a for a in range(4) if obs[4 + a] == 0]
            obs, reward, terminated, truncated, info = env.step(open_actions[step % len(open_actions)])
            if terminated or truncated:
                break


def test_partial_observability_same_obs_different_best_move():
    # maps 6 and 37 look the same at the start but need different first moves
    from src.astar import astar
    env = DroneNavEnv()
    obs6, _ = env.reset(options={"map_seed": 6})
    first6 = astar(env.grid, env.start, env.goal)[1]
    obs37, _ = env.reset(options={"map_seed": 37})
    first37 = astar(env.grid, env.start, env.goal)[1]
    assert np.array_equal(obs6, obs37)
    assert first6 == (0, 1)
    assert first37 == (1, 0)


def test_revisit_penalty_is_off_by_default():
    env = DroneNavEnv()
    env.reset(options={"map_seed": 42})
    env.step(1)
    obs, reward, *_ = env.step(3)
    assert reward == -1.0


def test_revisit_penalty_only_on_squares_already_visited():
    env = DroneNavEnv(revisit_penalty=2.0)
    env.reset(options={"map_seed": 42})
    obs, reward, *_ = env.step(1)
    assert reward == -1.0
    obs, reward, *_ = env.step(3)
    assert reward == -3.0
    obs, reward, *_ = env.step(0)
    assert reward == -30.0


def test_last_move_inputs_are_off_by_default():
    env = DroneNavEnv()
    obs, info = env.reset(options={"map_seed": 42})
    assert obs.shape == (8,)


def test_last_move_inputs_show_the_previous_move():
    env = DroneNavEnv(include_last_move=True)
    obs, info = env.reset(options={"map_seed": 42})
    assert obs.shape == (12,)
    assert list(obs[8:]) == [0, 0, 0, 0]
    obs, *_ = env.step(1)
    assert list(obs[8:]) == [0, 1, 0, 0]
    obs, *_ = env.step(3)
    assert list(obs[8:]) == [0, 0, 0, 1]


def test_last_move_env_passes_check_env():
    check_env(DroneNavEnv(include_last_move=True))


def test_gusts_are_off_by_default():
    assert DroneNavEnv().gust_prob == 0.0


def test_full_gusts_always_change_the_requested_move():
    env = DroneNavEnv(gust_prob=1.0)
    env.reset(options={"map_seed": 30005})
    for i in range(100):
        requested = i % 4
        executed = env._apply_gust(requested)
        assert executed != requested and executed in {0, 1, 2, 3}


def test_no_gusts_never_change_the_requested_move():
    env = DroneNavEnv(gust_prob=0.0)
    env.reset(options={"map_seed": 30005})
    for i in range(100):
        assert env._apply_gust(i % 4) == i % 4


def test_same_map_and_strength_give_the_same_gusts():
    actions = [1, 2] * 7
    flights = []
    for _ in range(2):
        env = DroneNavEnv(gust_prob=0.10)
        env.reset(options={"map_seed": 30005})
        executed = []
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(a)
            executed.append(info["executed_action"])
            if terminated or truncated:
                break
        flights.append((executed, env.pos))
    assert flights[0] == flights[1]


def test_gusts_happen_about_as_often_as_p_and_pick_the_other_moves_evenly():
    env = DroneNavEnv(gust_prob=0.10)
    env.reset(options={"map_seed": 30005})
    changed = Counter()
    gusts = 0
    for i in range(10_000):
        requested = i % 4
        executed = env._apply_gust(requested)
        if executed != requested:
            gusts += 1
            changed[(executed - requested) % 4] += 1
    assert 900 <= gusts <= 1100                          # about 10%
    assert all(250 <= changed[d] <= 420 for d in (1, 2, 3))  # each of the other 3 about a third


def test_a_gust_uses_the_executed_move_for_the_outcome():
    env = DroneNavEnv(gust_prob=1.0)
    env.reset(options={"map_seed": 30005})
    start = env.pos
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["executed_action"] != 0
    expected = step_outcome(env.grid, start, info["executed_action"], env.goal)
    assert (env.pos, reward, terminated, info["event"]) == expected


def test_gust_env_passes_check_env():
    check_env(DroneNavEnv(gust_prob=0.10, include_last_move=True))


def test_last_move_inputs_show_the_move_that_actually_happened():
    env = DroneNavEnv(gust_prob=1.0, include_last_move=True)
    env.reset(options={"map_seed": 30005})
    obs, reward, terminated, truncated, info = env.step(1)
    assert info["executed_action"] != 1
    if not terminated:
        assert list(obs[8:]) == [1.0 if a == info["executed_action"] else 0.0 for a in range(4)]


def open_map_env(**kwargs):
    # a wind env on an empty 8x8 map (no walls), so a test controls every square
    env = DroneNavEnv(include_last_move=True, **kwargs)
    env.reset(options={"map_seed": 42})
    env.grid = np.zeros((8, 8), dtype=env.grid.dtype)
    return env


def test_drift_onto_the_goal_is_a_goal():
    env = open_map_env(wind=(1, 1.0), drift_share=1.0)
    env.pos = (7, 5)
    _, reward, terminated, _, info = env.step(1)          # asks East to (7, 6), pushed East to (7, 7)
    assert env.pos == (7, 7) and reward == 100 and terminated and info["event"] == "goal"


def test_drift_into_a_wall_is_a_crash():
    env = open_map_env(wind=(1, 1.0), drift_share=1.0)
    env.grid[3, 5] = 1
    env.pos = (3, 3)
    _, reward, terminated, _, info = env.step(1)          # asks East to (3, 4), pushed East into the wall
    assert env.pos == (3, 4) and reward == -30 and terminated and info["event"] == "collision"


def test_wind_replacing_the_first_move_off_the_map_is_a_crash():
    env = DroneNavEnv(wind=(0, 1.0), drift_share=0.0)
    env.reset(options={"map_seed": 42})
    _, reward, terminated, _, info = env.step(2)          # asks South, wind makes it North, off the map
    assert info["executed_action"] == 0
    assert env.pos == (0, 0) and reward == -30 and terminated


def test_drift_against_the_move_ends_where_it_started():
    env = open_map_env(wind=(1, 1.0), drift_share=1.0)
    env.pos = (3, 3)
    obs, reward, terminated, _, _ = env.step(3)           # asks West to (3, 2), pushed East to (3, 3)
    assert env.pos == (3, 3) and reward == -1 and not terminated
    assert list(obs[8:12]) == [0, 1, 0, 0]


def test_wind_is_off_by_default():
    env = DroneNavEnv()
    obs, info = env.reset(options={"map_seed": 42})
    assert obs.shape == (8,) and info["wind"] is None
    for a in [1, 2, 1, 2]:
        _, _, terminated, _, info = env.step(a)
        assert info["wind"] is None and info["executed_action"] == a
        if terminated:
            break


def test_wind_aware_drone_has_17_inputs_ending_with_the_wind():
    env = DroneNavEnv(include_last_move=True, include_wind=True, wind=(2, 0.2))
    obs, _ = env.reset(options={"map_seed": 42})
    assert obs.shape == (17,)
    assert obs[-5:] == pytest.approx([0, 0, 1, 0, 0.2])


def test_zero_strength_never_changes_a_move():
    env = DroneNavEnv(wind=(1, 0.0))
    rng = np.random.default_rng(0)
    moves, seed = 0, 0
    while moves < 1000:
        env.reset(options={"map_seed": seed})
        seed = seed + 1
        terminated = truncated = False
        while not (terminated or truncated):
            a = int(rng.integers(4))
            _, _, terminated, truncated, info = env.step(a)
            assert info["wind"] is None and info["executed_action"] == a
            moves = moves + 1


def test_full_strength_replace_always_moves_with_the_wind():
    env = DroneNavEnv(wind=(3, 1.0), drift_share=0.0)
    for seed in range(20):
        env.reset(options={"map_seed": seed})
        _, _, _, _, info = env.step(1)
        assert info["wind"] == "replace" and info["executed_action"] == 3


def test_same_map_and_wind_give_the_same_wind_whatever_the_drone_does():
    effects = []
    for actions in ([1, 2] * 10, [2, 1] * 10):
        env = open_map_env(wind=(2, 0.3))
        effects.append([env.step(a)[4]["wind"] for a in actions[:4]])
    assert effects[0] == effects[1]


def test_drift_is_about_10_percent_of_wind_effects():
    env = DroneNavEnv(wind=(1, 1.0))
    env.reset(options={"map_seed": 42})
    effects = Counter(env._wind_effect() for _ in range(10_000))
    assert 800 <= effects["drift"] <= 1200
    assert effects["drift"] + effects["replace"] == 10_000


def test_training_schedule_uses_beta_short_blocks():
    env = DroneNavEnv(wind="schedule")
    winds = [env.reset()[1]["wind"] for _ in range(16)]
    assert winds[:4] == [(0, 0.0), (0, 0.1), (0, 0.2), (0, 0.3)]
    assert winds[4] == (1, 0.0)
    assert len(set(winds)) == 16


def test_wind_aware_env_passes_check_env():
    check_env(DroneNavEnv(include_last_move=True, include_wind=True, wind="schedule"))


def test_no_push_after_the_flight_has_already_ended():
    # added after a planted bug (pushing even after the drone's own move ended the flight) passed every test
    env = open_map_env(wind=(1, 1.0), drift_share=1.0)
    env.pos = (7, 6)
    _, reward, terminated, _, info = env.step(1)
    assert env.pos == (7, 7) and reward == 100 and terminated and info["event"] == "goal"


def test_diamond_flags_at_the_start_of_an_empty_map():
    env = DroneNavEnv(vision="diamond")
    env.reset(options={"map_seed": 42})
    env.grid = np.zeros((8, 8), dtype=env.grid.dtype)
    obs = env._get_obs()
    #                        N  E  S  W  NN EE SS WW NE SE SW NW
    assert list(obs[4:16]) == [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]


def test_diamond_sees_a_wall_two_squares_away_that_plus_cannot():
    for vision in ("plus", "diamond"):
        env = DroneNavEnv(vision=vision)
        env.reset(options={"map_seed": 42})
        env.grid = np.zeros((8, 8), dtype=env.grid.dtype)
        env.grid[5, 3] = 1
        env.grid[2, 4] = 1
        env.pos = (3, 3)
        obs = env._get_obs()
        if vision == "plus":
            assert list(obs[4:8]) == [0, 0, 0, 0]
        else:
            assert obs[4 + 6] == 1 and obs[4 + 8] == 1
            assert sum(obs[4:16]) == 2


def test_diamond_input_counts():
    assert DroneNavEnv(vision="diamond", include_last_move=True).observation_space.shape == (20,)
    assert DroneNavEnv(vision="diamond", include_last_move=True, include_wind=True,
                       wind=(0, 0.1)).reset(options={"map_seed": 1})[0].shape == (25,)


def test_diamond_env_passes_check_env():
    check_env(DroneNavEnv(vision="diamond", include_last_move=True, include_wind=True, wind="schedule"))


def shaped_env(**kwargs):
    env = open_map_env(shaped_reward=True, **kwargs)
    return env


def test_shaped_reward_is_off_by_default():
    env = open_map_env()
    _, reward, _, _, _ = env.step(1)
    assert reward == -1
    env.pos = (0, 0)
    _, reward, _, _, _ = env.step(0)
    assert reward == -30


def test_closer_is_plus_9_and_farther_is_minus_11():
    env = shaped_env()
    env.pos = (3, 3)
    _, closer, _, _, _ = env.step(2)          # South to (4, 3): 1 square closer
    _, farther, _, _, _ = env.step(0)         # North to (3, 3): 1 square farther
    assert closer == 9 and farther == -11


def test_a_closer_and_back_loop_costs_minus_2():
    env = shaped_env()
    env.pos = (3, 3)
    total = 0
    for a in [1, 3] * 10:
        total += env.step(a)[1]
    assert total == -20 and env.pos == (3, 3)


def test_crash_costs_more_than_a_full_timeout():
    env = shaped_env()
    _, reward, terminated, _, _ = env.step(0)
    assert reward == -201 and terminated       # max_steps 200 gives -(200 + 1)
    big = DroneNavEnv(size=16, max_steps=400, shaped_reward=True)
    big.reset(options={"map_seed": 60000})
    big.pos = (0, 0)
    assert big.step(3)[1] == -401


def test_reaching_the_goal_is_100_plus_10():
    env = shaped_env()
    env.pos = (7, 6)
    _, reward, terminated, _, info = env.step(1)
    assert reward == 110 and terminated and info["event"] == "goal"


def test_shaping_counts_the_wind_push_too():
    env = shaped_env(wind=(1, 1.0), drift_share=1.0)
    env.pos = (3, 3)
    _, reward, _, _, _ = env.step(1)          # East to (3, 4), pushed East to (3, 5): 2 closer
    assert env.pos == (3, 5) and reward == -1 + 20


def test_shaped_check_env():
    check_env(DroneNavEnv(size=16, max_steps=400, include_last_move=True, shaped_reward=True))


def test_visit_memory_is_off_by_default():
    assert DroneNavEnv(include_last_move=True).observation_space.shape == (12,)


def test_visit_memory_marks_squares_already_flown():
    env = DroneNavEnv(include_last_move=True, visit_memory=True)
    obs, _ = env.reset(options={"map_seed": 30000})
    assert env.observation_space.shape == (16,) and obs.shape == (16,)
    assert list(obs[-4:]) == [0, 0, 0, 0]            # nothing visited around the start yet (start itself is not in view)
    env.step(1)                                      # east to (0, 1); map 30000 is open there
    obs, _, _, _, _ = env.step(3)                    # back west to (0, 0)
    assert env.pos == (0, 0)
    assert list(obs[-4:]) == [0, 1, 0, 0]            # N, E, S, W: only East (0, 1) was visited


def test_visit_memory_env_passes_check_env():
    check_env(DroneNavEnv(include_last_move=True, visit_memory=True, vision="diamond", random_start_goal=True))
