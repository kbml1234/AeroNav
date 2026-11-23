"""Exploratory 16 x 16 probe: test run, run once.

Run:  python -m experiments.scale_test
"""
import csv
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from src.astar import astar
from src.astar_agent import run_astar_episode
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, route_efficiency, split_seeds

SIZE, MAX_STEPS = 16, 400
flights_file, summary_file = Path("results/06_scale16/scale_test_flights.csv"), Path("results/06_scale16/scale_test_summary.csv")
if flights_file.exists() or summary_file.exists():
    sys.exit("The 16 x 16 test has already been run. It is run once.")

maps = split_seeds("scale_test")
shortest = {}
for seed in maps:
    env = DroneNavEnv(size=SIZE, max_steps=MAX_STEPS)
    env.reset(options={"map_seed": seed})
    shortest[seed] = astar(env.grid, env.start, env.goal)

rows = []
env = DroneNavEnv(size=SIZE, max_steps=MAX_STEPS)
for seed in maps:
    f = run_astar_episode(env, seed)
    flight = {"ending": f["ending"], "steps": f["moves"]}
    rows.append({"drone": "A*", "train_seed": "", "map_seed": seed, "ending": f["ending"], "moves": f["moves"],
                 "efficiency": route_efficiency(flight, shortest[seed])})
for vision, name in [("plus", "S1 plus"), ("diamond", "S2 diamond")]:
    env = DroneNavEnv(size=SIZE, max_steps=MAX_STEPS, include_last_move=True, vision=vision)
    for s in range(3):
        model = DQN.load(f"results/models/scale16_{vision}_seed{s}/dqn_5000000_steps.zip")
        for seed in maps:
            f = fly_dqn(env, model, seed)
            rows.append({"drone": name, "train_seed": s, "map_seed": seed, "ending": f["ending"], "moves": f["steps"],
                         "efficiency": route_efficiency(f, shortest[seed])})

with open(flights_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for key in sorted({(r["drone"], str(r["train_seed"])) for r in rows}):
    group = [r for r in rows if (r["drone"], str(r["train_seed"])) == key]
    goals = [r for r in group if r["ending"] == "goal"]
    n = len(group)
    summary.append({"drone": key[0], "train_seed": key[1], "n": n, "success_%": round(100 * len(goals) / n, 1),
                    "crash_%": round(100 * sum(r["ending"] == "collision" for r in group) / n, 1),
                    "timeout_%": round(100 * sum(r["ending"] == "timeout" for r in group) / n, 1),
                    "moves_on_successes": round(float(np.mean([r["moves"] for r in goals])), 2) if goals else "",
                    "efficiency_on_successes": round(float(np.mean([r["efficiency"] for r in goals])), 3) if goals else ""})
with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print("saved", flights_file, "and", summary_file)
