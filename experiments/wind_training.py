"""Train the wind-aware and wind-blind DQNs.

Run:  python -m experiments.wind_training <aware|blind|aware_diamond|blind_diamond> <seed>
"""
import csv
import sys
import time
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dqn_agent import make_dqn
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

torch.set_num_threads(1)   # one CPU thread per run
TOTAL_STEPS = 500_000
EVERY = 50_000
VARIANTS = {"aware": (True, "plus"), "blind": (False, "plus"),
            "aware_diamond": (True, "diamond"), "blind_diamond": (False, "diamond")}
STRENGTHS = [0.0, 0.1, 0.2, 0.3]

variant, seed = sys.argv[1], int(sys.argv[2])
include_wind, vision = VARIANTS[variant]
tag = f"wind_{variant}_seed{seed}"
checkpoints = Path(f"results/models/{tag}")
checkpoints.mkdir(parents=True, exist_ok=True)

train = load_splits()["train"]
env = DroneNavEnv(seed_range=(train["low"], train["high"]), include_last_move=True,
                  include_wind=include_wind, wind="schedule", vision=vision)
env.reset(seed=seed)
model = make_dqn(env, seed=seed)
start = time.time()
model.learn(total_timesteps=TOTAL_STEPS,
            callback=CheckpointCallback(save_freq=EVERY, save_path=str(checkpoints), name_prefix="dqn"))
minutes = (time.time() - start) / 60
print(f"{tag}: trained in {minutes:.1f} minutes")

rows = []
for d in range(4):
    for s in STRENGTHS:
        eval_env = DroneNavEnv(include_last_move=True, include_wind=include_wind, wind=(d, s), vision=vision)
        r = summarise([fly_dqn(eval_env, model, m) for m in split_seeds("validation")])
        rows.append({"variant": variant, "seed": seed, "direction": "NESW"[d], "strength_%": int(100 * s),
                     "n": r["n"], "success_%": round(100 * r["success_rate"], 1),
                     "crashes": r["collisions"], "timeouts": r["timeouts"], "train_minutes": round(minutes, 1)})
        print(rows[-1])

with open(f"results/{'05_vision' if 'diamond' in tag else '04_wind'}/wind_validation_check_{tag}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
