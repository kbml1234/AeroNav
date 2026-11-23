"""Random start/goal follow-up: the start and goal rule."""
from gymnasium.utils.env_checker import check_env

from src.astar import astar
from src.drone_env import DroneNavEnv
from src.map_generator import generate_map, generate_trip_map, min_route_length


def test_min_route_length_matches_the_protocol_table():
    assert [min_route_length(s) for s in (8, 16, 32, 64, 100)] == [10, 21, 43, 88, 139]


def test_rule_holds_on_many_maps():
    for size in (8, 16):
        need = min_route_length(size)
        for seed in range(200):
            grid, start, goal = generate_trip_map(size, 0.15, seed)
            assert grid[start] == 0 and grid[goal] == 0
            route = astar(grid, start, goal)
            assert route is not None
            assert len(route) - 1 >= need      # moves = squares - 1


def test_same_seed_gives_the_same_trip():
    a = generate_trip_map(8, 0.15, 70000)
    b = generate_trip_map(8, 0.15, 70000)
    assert (a[0] == b[0]).all() and a[1] == b[1] and a[2] == b[2]


def test_starts_and_goals_really_vary():
    trips = {generate_trip_map(8, 0.15, seed)[1:] for seed in range(100)}
    starts = {s for s, _ in trips}
    goals = {g for _, g in trips}
    assert len(starts) > 20 and len(goals) > 20


def test_default_env_still_flies_corner_to_corner():
    env = DroneNavEnv()
    env.reset(options={"map_seed": 30000})
    grid, start, goal = generate_map(8, 0.15, 30000)
    assert env.start == (0, 0) == start and env.goal == (7, 7) == goal
    assert (env.grid == grid).all()


def test_env_uses_the_trip_map_and_tells_the_drone_the_goal():
    env = DroneNavEnv(random_start_goal=True, include_last_move=True)
    obs, _ = env.reset(options={"map_seed": 70000})
    grid, start, goal = generate_trip_map(8, 0.15, 70000)
    assert (env.grid == grid).all() and env.start == start and env.goal == goal
    assert env.pos == start
    assert list(obs[:4]) == [start[0] / 7, start[1] / 7, goal[0] / 7, goal[1] / 7]


def test_env_passes_the_gymnasium_check():
    check_env(DroneNavEnv(random_start_goal=True, include_last_move=True))
