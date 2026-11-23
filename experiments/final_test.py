"""Final test run, run once.

Run:  python -m experiments.final_test
python -m experiments.final_test --dry-run   (20 validation maps), writes to the scratch folder given
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

GUSTS = [0.0, 0.05, 0.10, 0.20]
DQNS = {("calm DQN", 0): "results/models/final/dqn_calm_last_move_500k.zip",
        ("calm DQN", 1): "results/models/curve_last_move_seed1/dqn_500000_steps.zip",
        ("calm DQN", 2): "results/models/curve_last_move_seed2/dqn_500000_steps.zip",
        ("gust DQN", 0): "results/models/curve_gust/dqn_500000_steps.zip",
        ("gust DQN", 1): "results/models/curve_gust_seed1/dqn_500000_steps.zip",
        ("gust DQN", 2): "results/models/curve_gust_seed2/dqn_500000_steps.zip"}

dry_run = "--dry-run" in sys.argv
if dry_run:
    maps = split_seeds("validation")[:20]
    out_dir = Path(sys.argv[sys.argv.index("--dry-run") + 1])
else:
    maps = split_seeds("test")
    out_dir = Path("results/03_main_test")
flights_file, summary_file = out_dir / "final_test_flights.csv", out_dir / "final_test_summary.csv"
if not dry_run and (flights_file.exists() or summary_file.exists()):
    sys.exit("The final test has already been run. It is run once (protocol section 7).")

shortest = {}
for seed in maps:
    env = DroneNavEnv()
    env.reset(options={"map_seed": seed})
    shortest[seed] = astar(env.grid, env.start, env.goal)

rows = []
for p in GUSTS:
    env = DroneNavEnv(gust_prob=p)
    for seed in maps:
        f = run_astar_episode(env, seed)
        flight = {"ending": f["ending"], "steps": f["moves"]}
        rows.append({"agent": "A*", "train_seed": "", "gust_%": int(100 * p), "map_seed": seed,
                     "ending": f["ending"], "moves": f["moves"], "efficiency": route_efficiency(flight, shortest[seed])})
    for (name, train_seed), path in DQNS.items():
        model = DQN.load(path)
        env = DroneNavEnv(include_last_move=True, gust_prob=p)
        for seed in maps:
            f = fly_dqn(env, model, seed)
            rows.append({"agent": name, "train_seed": train_seed, "gust_%": int(100 * p), "map_seed": seed,
                         "ending": f["ending"], "moves": f["steps"], "efficiency": route_efficiency(f, shortest[seed])})
    print(f"gust {int(100 * p)}% done")

with open(flights_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for key in sorted({(r["agent"], r["train_seed"], r["gust_%"]) for r in rows}, key=lambda k: (k[0], str(k[1]), k[2])):
    group = [r for r in rows if (r["agent"], r["train_seed"], r["gust_%"]) == key]
    goals = [r for r in group if r["ending"] == "goal"]
    n = len(group)
    summary.append({"agent": key[0], "train_seed": key[1], "gust_%": key[2], "n": n,
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
