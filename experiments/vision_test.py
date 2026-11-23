"""Diamond-vision test run, run once.

Run:  python -m experiments.vision_test
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

STRENGTHS = [0.0, 0.1, 0.2, 0.3]
DRONES = {
    "A2 diamond+wind": (True, "diamond", {s: f"results/models/wind_aware_diamond_seed{s}/dqn_500000_steps.zip" for s in range(3)}),
    "B2 diamond": (False, "diamond", {s: f"results/models/wind_blind_diamond_seed{s}/dqn_500000_steps.zip" for s in range(3)}),
    "A wind-aware": (True, "plus", {s: f"results/models/wind_aware_seed{s}/dqn_500000_steps.zip" for s in range(3)}),
    "B wind-blind": (False, "plus", {s: f"results/models/wind_blind_seed{s}/dqn_500000_steps.zip" for s in range(3)}),
    "C calm": (False, "plus", {0: "results/models/final/dqn_calm_last_move_500k.zip",
                               1: "results/models/curve_last_move_seed1/dqn_500000_steps.zip",
                               2: "results/models/curve_last_move_seed2/dqn_500000_steps.zip"}),
}

flights_file, summary_file = Path("results/05_vision/vision_test_flights.csv"), Path("results/05_vision/vision_test_summary.csv")
if flights_file.exists() or summary_file.exists():
    sys.exit("The vision test has already been run. It is run once.")

maps = split_seeds("vision_test")
shortest = {}
for seed in maps:
    env = DroneNavEnv()
    env.reset(options={"map_seed": seed})
    shortest[seed] = astar(env.grid, env.start, env.goal)

models = {(name, s): DQN.load(path) for name, (_, _, paths) in DRONES.items() for s, path in paths.items()}
rows = []
for d in range(4):
    for strength in STRENGTHS:
        wind = (d, strength)
        condition = {"direction": "NESW"[d], "strength_%": int(100 * strength)}
        env = DroneNavEnv(wind=wind)
        for seed in maps:
            f = run_astar_episode(env, seed)
            flight = {"ending": f["ending"], "steps": f["moves"]}
            rows.append({"drone": "D A*", "train_seed": "", **condition, "map_seed": seed, "ending": f["ending"],
                         "moves": f["moves"], "efficiency": route_efficiency(flight, shortest[seed])})
        for (name, s), model in models.items():
            env = DroneNavEnv(include_last_move=True, include_wind=DRONES[name][0], wind=wind,
                              vision=DRONES[name][1])
            for seed in maps:
                f = fly_dqn(env, model, seed)
                rows.append({"drone": name, "train_seed": s, **condition, "map_seed": seed, "ending": f["ending"],
                             "moves": f["steps"], "efficiency": route_efficiency(f, shortest[seed])})
        print("done:", condition)

with open(flights_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary = []
keys = sorted({(r["drone"], str(r["train_seed"]), r["direction"], r["strength_%"]) for r in rows})
for key in keys:
    group = [r for r in rows if (r["drone"], str(r["train_seed"]), r["direction"], r["strength_%"]) == key]
    goals = [r for r in group if r["ending"] == "goal"]
    n = len(group)
    summary.append({"drone": key[0], "train_seed": key[1], "direction": key[2], "strength_%": key[3], "n": n,
                    "success_%": round(100 * len(goals) / n, 1),
                    "crash_%": round(100 * sum(r["ending"] == "collision" for r in group) / n, 1),
                    "timeout_%": round(100 * sum(r["ending"] == "timeout" for r in group) / n, 1),
                    "moves_on_successes": round(float(np.mean([r["moves"] for r in goals])), 2) if goals else "",
                    "efficiency_on_successes": round(float(np.mean([r["efficiency"] for r in goals])), 3) if goals else ""})
with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print("saved", flights_file, "and", summary_file)
