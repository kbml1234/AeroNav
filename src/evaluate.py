import json

import numpy as np

SPLITS_FILE = "scenarios/splits.json"


def load_splits(filename=SPLITS_FILE):
    with open(filename) as f:
        return json.load(f)


def split_seeds(name, filename=SPLITS_FILE):
    split = load_splits(filename)[name]
    return list(range(split["low"], split["high"]))


def fly_dqn(env, model, map_seed):
    # deterministic: always the top Q-value
    obs, info = env.reset(options={"map_seed": map_seed})
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            ending = info["event"] if terminated else "timeout"
            return {"map_seed": map_seed, "ending": ending, "steps": env.steps}


def summarise(results):
    n = len(results)
    endings = [r["ending"] for r in results]
    goal_steps = [r["steps"] for r in results if r["ending"] == "goal"]
    return {"n": n,
            "success_rate": endings.count("goal") / n,
            "collisions": endings.count("collision"),
            "timeouts": endings.count("timeout"),
            "mean_goal_steps": float(np.mean(goal_steps)) if goal_steps else None}


def route_efficiency(flight, route):
    if flight["ending"] != "goal" or route is None:
        return None                      # a crash or timeout has no efficiency (a 2-move crash is not "700%")
    shortest_moves = len(route) - 1      # the route counts squares; moves = squares - 1
    return shortest_moves / flight["steps"]
