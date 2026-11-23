"""Random start/goal follow-up: test run, run once.

Run:  python -m experiments.startgoal_test
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

DRONES = {
    "C corner-trained": {0: "results/models/final/dqn_calm_last_move_500k.zip",
                         1: "results/models/curve_last_move_seed1/dqn_500000_steps.zip",
                         2: "results/models/curve_last_move_seed2/dqn_500000_steps.zip"},
    "R random-trained, reward fix": {s: f"results/models/startgoal_fix_seed{s}/dqn_500000_steps.zip" for s in range(3)},
    "R0 random-trained, original reward": {s: f"results/models/startgoal_seed{s}/dqn_500000_steps.zip" for s in range(3)},
}
PARTS = [("random start/goal", "startgoal_test", True), ("corner check", "test", False)]

flights_file, summary_file = Path("results/10_startgoal/startgoal_test_flights.csv"), Path("results/10_startgoal/startgoal_test_summary.csv")
if flights_file.exists() or summary_file.exists():
    sys.exit("The start/goal test has already been run. It is run once.")

models = {(name, s): DQN.load(path) for name, paths in DRONES.items() for s, path in paths.items()}
rows = []
for part, split, random_trips in PARTS:
    maps = split_seeds(split)
    env = DroneNavEnv(include_last_move=True, random_start_goal=random_trips)
    shortest = {}
    for seed in maps:
        env.reset(options={"map_seed": seed})
        shortest[seed] = astar(env.grid, env.start, env.goal)
    if random_trips:
        for seed in maps:
            f = run_astar_episode(env, seed)
            rows.append({"part": part, "drone": "A*", "train_seed": "", "map_seed": seed, "ending": f["ending"],
                         "moves": f["moves"], "shortest": len(shortest[seed]) - 1,
                         "efficiency": route_efficiency({"ending": f["ending"], "steps": f["moves"]}, shortest[seed])})
    for (name, s), model in models.items():
        for seed in maps:
            f = fly_dqn(env, model, seed)
            rows.append({"part": part, "drone": name, "train_seed": s, "map_seed": seed, "ending": f["ending"],
                         "moves": f["steps"], "shortest": len(shortest[seed]) - 1,
                         "efficiency": route_efficiency(f, shortest[seed])})
    print("done:", part, flush=True)

with open(flights_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for key in sorted({(r["part"], r["drone"], str(r["train_seed"])) for r in rows}):
    group = [r for r in rows if (r["part"], r["drone"], str(r["train_seed"])) == key]
    goals = [r for r in group if r["ending"] == "goal"]
    n = len(group)
    summary.append({"part": key[0], "drone": key[1], "train_seed": key[2], "n": n,
                    "success_%": round(100 * len(goals) / n, 1),
                    "crash_%": round(100 * sum(r["ending"] == "collision" for r in group) / n, 1),
                    "timeout_%": round(100 * sum(r["ending"] == "timeout" for r in group) / n, 1),
                    "moves_on_successes": round(float(np.mean([r["moves"] for r in goals])), 2) if goals else "",
                    "efficiency_on_successes": round(float(np.mean([r["efficiency"] for r in goals])), 3) if goals else ""})
with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
