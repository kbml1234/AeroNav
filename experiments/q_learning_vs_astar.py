"""A* vs tabular Q-learning, starting on the same fixed map (seed 42).

Run:  python -m experiments.q_learning_vs_astar
"""
import csv
import json
from pathlib import Path

import numpy as np

from src.astar import astar
from src.astar_agent import run_astar_episode
from src.drone_env import DroneNavEnv
from src.q_learning import run_greedy, train

SIZE = 8
DENSITY = 0.15
MAP_SEED = 42
ALPHA = 0.1
GAMMA = 0.95
BUDGETS = [250, 500, 1000, 2000]
TRAIN_SEEDS = range(10)
NEW_MAP_SEEDS = range(10_000, 10_300)

Path("results").mkdir(exist_ok=True)
with open("scenarios/q_learning_baseline.json", "w") as f:
    json.dump({"size": SIZE, "density": DENSITY, "map_seed": MAP_SEED, "alpha": ALPHA, "gamma": GAMMA,
               "epsilon": {"start": 1.0, "end": 0.05, "decay_fraction": 0.8},
               "day11_episodes": 2000, "day11_seed": 0,
               "budgets": BUDGETS, "train_seeds": list(TRAIN_SEEDS),
               "new_map_seeds": list(NEW_MAP_SEEDS)}, f, indent=2)

env = DroneNavEnv(size=SIZE, obstacle_density=DENSITY)
env.reset(options={"map_seed": MAP_SEED})
shortest = len(astar(env.grid, env.start, env.goal)) - 1

rows = []
for budget in BUDGETS:
    for seed in TRAIN_SEEDS:
        q_table, history = train(env, MAP_SEED, budget, ALPHA, GAMMA, seed=seed)
        final = run_greedy(env, q_table, MAP_SEED)
        rows.append({"budget": budget, "seed": seed,
                     "train_moves": sum(row["steps"] for row in history),
                     "train_crashes": sum(row["ending"] == "collision" for row in history),
                     "final_ending": final["ending"], "final_steps": final["steps"],
                     "shortest": final["ending"] == "goal" and final["steps"] == shortest})

with open("results/01_baselines/q_learning_vs_astar_reliability.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"map {MAP_SEED}: A* shortest = {shortest} moves, 0 training rounds, 0 crashes (but it is given the whole map)\n")
print("Q-learning, 10 training seeds per budget:")
for budget in BUDGETS:
    runs = [r for r in rows if r["budget"] == budget]
    goals = sum(r["final_ending"] == "goal" for r in runs)
    best = sum(r["shortest"] for r in runs)
    crashes = np.mean([r["train_crashes"] for r in runs])
    moves = np.mean([r["train_moves"] for r in runs])
    print(f"  {budget:5d} rounds: reaches goal {goals}/10, shortest route {best}/10, "
          f"avg {moves:,.0f} training moves, avg {crashes:,.0f} training crashes")

q_table, _ = train(env, MAP_SEED, 2000, ALPHA, GAMMA, seed=0)
new_rows = []
for map_seed in NEW_MAP_SEEDS:
    q = run_greedy(env, q_table, map_seed)
    a = run_astar_episode(env, map_seed)
    new_rows.append({"map_seed": map_seed, "q_ending": q["ending"], "q_steps": q["steps"],
                     "astar_ending": a["ending"], "astar_steps": a["moves"]})

with open("results/01_baselines/q_learning_vs_astar_new_maps.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
    writer.writeheader()
    writer.writerows(new_rows)

n = len(new_rows)
print(f"\n{n} new maps, map-42 Q-table vs A*:")
for name in ["q", "astar"]:
    endings = [r[f"{name}_ending"] for r in new_rows]
    counts = {e: endings.count(e) for e in ["goal", "collision", "timeout"]}
    print(f"  {'Q-learning' if name == 'q' else 'A*':10s} goal {counts['goal']:3d} ({100 * counts['goal'] / n:.0f}%)  "
          f"crash {counts['collision']:3d}  timeout {counts['timeout']:3d}")
