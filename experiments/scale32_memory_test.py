"""32 x 32 visited-square memory: test run, run once, best validation checkpoint.

Run:  python -m experiments.scale32_memory_test
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


def best(curve, model_dir):
    # highest validation success; a tie goes to the earlier checkpoint
    c = pd.read_csv(curve)
    row = c.loc[c["success_%"].idxmax()]
    return f"{model_dir}/dqn_{int(row['train_moves'])}_steps.zip"


DRONES = {  # name: (visit_memory, {seed: model file})
    "M memory": (True, {s: best(f"results/14_memory/scale32memory_seed{s}_curve.csv", f"results/models/scale32memory_seed{s}")
                        for s in range(3)}),
    "K no memory": (False, {s: best(f"results/12_scale32_long/scale32long_K_seed{s}_curve.csv", f"results/models/scale32long_K_seed{s}")
                            for s in range(3)}),
}
for name, (_, paths) in DRONES.items():
    print(name, paths)

out = Path("results/14_memory")
flights_file, summary_file = out / "memory_test_flights.csv", out / "memory_test_summary.csv"
if flights_file.exists() or summary_file.exists():
    sys.exit("The memory test has already been run. It is run once.")

maps = split_seeds("memory_test")
envs = {flag: DroneNavEnv(size=32, max_steps=800, include_last_move=True, vision="diamond", random_start_goal=True,
                          visit_memory=flag) for flag in (True, False)}
shortest = {}
for seed in maps:
    envs[False].reset(options={"map_seed": seed})
    shortest[seed] = astar(envs[False].grid, envs[False].start, envs[False].goal)

rows = []
for seed in maps:
    f = run_astar_episode(envs[False], seed)
    rows.append({"drone": "A*", "train_seed": "", "map_seed": seed, "ending": f["ending"], "moves": f["moves"],
                 "shortest": len(shortest[seed]) - 1,
                 "efficiency": route_efficiency({"ending": f["ending"], "steps": f["moves"]}, shortest[seed])})
for name, (memory, paths) in DRONES.items():
    for s, path in paths.items():
        model = DQN.load(path)
        for seed in maps:
            f = fly_dqn(envs[memory], model, seed)
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
