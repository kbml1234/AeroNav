from src.astar import astar
from src.drone_env import ACTIONS

# (row change, col change) to action number
DIRECTION_TO_ACTION = {change: action for action, change in ACTIONS.items()}


def next_action(grid, pos, goal):
    # replan every step so A* recovers after a gust
    route = astar(grid, pos, goal)
    if route is None or len(route) < 2:
        return None
    (r1, c1), (r2, c2) = route[0], route[1]
    return DIRECTION_TO_ACTION[(r2 - r1, c2 - c1)]


def run_astar_episode(env, map_seed):
    env.reset(options={"map_seed": map_seed})
    while True:
        action = next_action(env.grid, env.pos, env.goal)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            ending = info["event"] if terminated else "timeout"
            return {"moves": env.steps, "ending": ending}
