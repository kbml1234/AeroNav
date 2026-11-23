"""Learning curves and the learned route, from the training run.

python -m experiments.q_learning_curves
"""
import csv
from pathlib import Path

import numpy as np

from src.astar import astar
from src.drone_env import DroneNavEnv
from src.learning_curves import plot_learning_curves
from src.q_learning import choose_action
from src.render import record_episode, save_gif, save_png

MAP_SEED = 42
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

with open("results/01_baselines/q_learning_fixed_map.csv") as f:
    history = [{"episode": int(row["episode"]), "reward": float(row["reward"]), "steps": int(row["steps"]),
                "ending": row["ending"], "epsilon": float(row["epsilon"])} for row in csv.DictReader(f)]
q_table = np.load("results/01_baselines/q_table_fixed_map.npy")

env = DroneNavEnv(size=8, obstacle_density=0.15)
env.reset(options={"map_seed": MAP_SEED})
shortest = len(astar(env.grid, env.start, env.goal)) - 1

plot_learning_curves(history, shortest, OUT / "day12_learning_curves.png")

rng = np.random.default_rng(0)
env.reset(options={"map_seed": MAP_SEED})
actions = []
while True:
    action = choose_action(q_table, env.pos, 0.0, rng)
    actions.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
route = record_episode(env, actions, options={"map_seed": MAP_SEED})
save_png(route, OUT / "day12_q_learning_route.png")
save_gif(route, OUT / "day12_q_learning_route.gif")
print(f"learned route: {route['ending']} in {route['steps']} steps (A* shortest: {shortest})")
print("saved", OUT / "day12_learning_curves.png", "and the route picture")
