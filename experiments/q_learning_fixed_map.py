"""Train tabular Q-learning on one fixed 8x8 map.

Run:  python -m experiments.q_learning_fixed_map
"""
import csv
from pathlib import Path

import numpy as np

from src.astar import astar
from src.drone_env import DroneNavEnv
from src.q_learning import run_greedy, train

SIZE = 8
DENSITY = 0.15
MAP_SEED = 42
EPISODES = 2000
ALPHA = 0.1
GAMMA = 0.95
SEED = 0

Path("results").mkdir(exist_ok=True)

env = DroneNavEnv(size=SIZE, obstacle_density=DENSITY)
q_table, history = train(env, MAP_SEED, EPISODES, ALPHA, GAMMA, seed=SEED)

with open("results/01_baselines/q_learning_fixed_map.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)
np.save("results/01_baselines/q_table_fixed_map.npy", q_table)

for start in range(0, EPISODES, 200):
    block = history[start:start + 200]
    goals = sum(row["ending"] == "goal" for row in block)
    crashes = sum(row["ending"] == "collision" for row in block)
    steps = [row["steps"] for row in block if row["ending"] == "goal"]
    avg_steps = f"{sum(steps) / len(steps):.1f}" if steps else "-"
    print(f"rounds {start:4d}-{start + 199:4d}  eps {block[0]['epsilon']:.2f}  "
          f"goal {goals:3d}/200  crash {crashes:3d}  avg steps when goal {avg_steps}")

env.reset(options={"map_seed": MAP_SEED})
shortest = len(astar(env.grid, env.start, env.goal)) - 1
final = run_greedy(env, q_table, MAP_SEED)
print(f"\nfinal flight with epsilon = 0: {final['ending']} in {final['steps']} steps "
      f"(A* shortest: {shortest})")
