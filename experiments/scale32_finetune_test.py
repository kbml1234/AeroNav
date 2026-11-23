"""32 x 32 fine-tuning: test run, run once, best validation checkpoint.

Run:  python -m experiments.scale32_finetune_test
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from src.astar import astar
from src.astar_agent import run_astar_episode
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, route_efficiency, split_seeds



def best_finetuned(seed):
    # highest validation success among the 8 fine-tuned checkpoints; a tie goes to the earlier one
    c = pd.read_csv(f"results/13_scale32_finetune/scale32finetune_seed{seed}_curve.csv")
    row = c.loc[c["success_%"].idxmax()]
    return f"results/models/scale32finetune_seed{seed}/dqn_plus{int(row['extra_moves'])}_steps.zip"


LONG_K = {0: 20_000_000, 1: 22_000_000, 2: 14_000_000}
DRONES = {"F fine-tuned": {s: best_finetuned(s) for s in range(3)},
          "K curriculum (before fine-tuning)": {s: f"results/models/scale32long_K_seed{s}/dqn_{LONG_K[s]}_steps.zip" for s in range(3)}}
for name, paths in DRONES.items():
    print(name, paths)

flights_file, summary_file = Path("results/13_scale32_finetune/scale32finetune_test_flights.csv"), Path("results/13_scale32_finetune/scale32finetune_test_summary.csv")
if flights_file.exists() or summary_file.exists():
    sys.exit("The 32 x 32 test has already been run. It is run once.")

maps = split_seeds("scale32_finetune_test")
env = DroneNavEnv(size=32, max_steps=800, include_last_move=True, vision="diamond", random_start_goal=True)
shortest = {}
for seed in maps:
    env.reset(options={"map_seed": seed})
    shortest[seed] = astar(env.grid, env.start, env.goal)

rows = []
for seed in maps:
    f = run_astar_episode(env, seed)
    rows.append({"drone": "A*", "train_seed": "", "map_seed": seed, "ending": f["ending"], "moves": f["moves"],
                 "shortest": len(shortest[seed]) - 1,
                 "efficiency": route_efficiency({"ending": f["ending"], "steps": f["moves"]}, shortest[seed])})
for name, paths in DRONES.items():
    for s, path in paths.items():
        model = DQN.load(path)
        for seed in maps:
            f = fly_dqn(env, model, seed)
            rows.append({"drone": name, "train_seed": s, "map_seed": seed, "ending": f["ending"], "moves": f["steps"],
                         "shortest": len(shortest[seed]) - 1, "efficiency": route_efficiency(f, shortest[seed])})
        print("done:", name, "seed", s, flush=True)

with open(flights_file, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for key in sorted({(r["drone"], str(r["train_seed"])) for r in rows}):
    group = [r for r in rows if (r["drone"], str(r["train_seed"])) == key]
    goals = [r for r in group if r["ending"] == "goal"]
    n = len(group)
    summary.append({"drone": key[0], "train_seed": key[1], "n": n,
                    "success_%": round(100 * len(goals) / n, 1),
                    "crash_%": round(100 * sum(r["ending"] == "collision" for r in group) / n, 1),
                    "timeout_%": round(100 * sum(r["ending"] == "timeout" for r in group) / n, 1),
                    "moves_on_successes": round(float(np.mean([r["moves"] for r in goals])), 2) if goals else "",
                    "efficiency_on_successes": round(float(np.mean([r["efficiency"] for r in goals])), 3) if goals else ""})
with open(summary_file, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
