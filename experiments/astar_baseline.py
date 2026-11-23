"""A* baseline on 100 static, solvable maps at two difficulty levels.

Run:  python -m experiments.astar_baseline
"""
import csv
import json
from pathlib import Path

from src.astar_agent import run_astar_episode
from src.drone_env import DroneNavEnv
from src.grid_utils import manhattan

SIZE = 12
DENSITIES = {"easy": 0.15, "hard": 0.30}
MAP_SEEDS = list(range(100))

Path("results").mkdir(exist_ok=True)
Path("scenarios").mkdir(exist_ok=True)

with open("scenarios/astar_baseline.json", "w") as f:
    json.dump({"size": SIZE, "densities": DENSITIES, "map_seeds": MAP_SEEDS,
               "generator": "src/map_generator.py generate_map"}, f, indent=2)

rows = []
for level, density in DENSITIES.items():
    env = DroneNavEnv(size=SIZE, obstacle_density=density)
    for map_seed in MAP_SEEDS:
        result = run_astar_episode(env, map_seed)
        straight_line = manhattan(env.start, env.goal)
        rows.append({"level": level, "map_seed": map_seed, "size": SIZE, "density": density,
                     "moves": result["moves"], "extra_moves": result["moves"] - straight_line,
                     "ending": result["ending"]})

with open("results/01_baselines/astar_baseline.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

for level in DENSITIES:
    group = [row for row in rows if row["level"] == level]
    moves = [row["moves"] for row in group]
    goals = sum(row["ending"] == "goal" for row in group)
    detours = sum(row["extra_moves"] > 0 for row in group)
    print(f"{level} ({DENSITIES[level]:.0%} walls): reached goal {goals}/{len(group)}, "
          f"moves average {sum(moves) / len(moves):.1f} (fewest {min(moves)}, most {max(moves)}), "
          f"maps needing a detour {detours}")
print("saved results/01_baselines/astar_baseline.csv and scenarios/astar_baseline.json")
