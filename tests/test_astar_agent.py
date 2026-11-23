from src.astar import astar
from src.astar_agent import next_action, run_astar_episode
from src.drone_env import DroneNavEnv


def test_next_action_follows_the_route():
    env = DroneNavEnv()
    env.reset(seed=42)
    route = astar(env.grid, env.pos, env.goal)
    action = next_action(env.grid, env.pos, env.goal)
    env.step(action)
    assert env.pos == route[1]


def test_no_action_on_the_goal():
    env = DroneNavEnv()
    env.reset(seed=42)
    assert next_action(env.grid, env.goal, env.goal) is None


def test_astar_flies_the_shortest_route():
    env = DroneNavEnv(size=12)
    for map_seed in range(20):
        result = run_astar_episode(env, map_seed)
        shortest = len(astar(env.grid, env.start, env.goal)) - 1
        assert result["ending"] == "goal"
        assert result["moves"] == shortest


def test_replans_after_being_pushed_off_course():
    env = DroneNavEnv(size=12)
    env.reset(options={"map_seed": 3})
    for _ in range(3):
        env.step(next_action(env.grid, env.pos, env.goal))
    r, c = env.pos
    pushed = next(p for p in [(r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1)]
                  if 0 <= p[0] < 12 and 0 <= p[1] < 12 and env.grid[p] == 0)
    env.pos = pushed
    expected_moves = len(astar(env.grid, pushed, env.goal)) - 1
    moves = 0
    while env.pos != env.goal:
        env.step(next_action(env.grid, env.pos, env.goal))
        moves += 1
    assert moves == expected_moves
