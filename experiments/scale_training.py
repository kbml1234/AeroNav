"""Exploratory 16 x 16 probe: train one DQN on 16 x 16 maps.

Run:  python -m experiments.scale_training <plus|diamond> <seed>
"""
import csv
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dqn_agent import make_dqn
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

torch.set_num_threads(1)
SIZE, MAX_STEPS = 16, 400
TOTAL_STEPS = 5_000_000
EVERY = 500_000

vision, seed = sys.argv[1], int(sys.argv[2])
tag = f"scale16_{vision}_seed{seed}"
checkpoints = Path(f"results/models/{tag}")
checkpoints.mkdir(parents=True, exist_ok=True)

train = load_splits()["train"]
env = DroneNavEnv(size=SIZE, max_steps=MAX_STEPS, seed_range=(train["low"], train["high"]),
                  include_last_move=True, vision=vision)
env.reset(seed=seed)
model = make_dqn(env, seed=seed)
start = time.time()
model.learn(total_timesteps=TOTAL_STEPS,
            callback=CheckpointCallback(save_freq=EVERY, save_path=str(checkpoints), name_prefix="dqn"))
minutes = (time.time() - start) / 60
print(f"{tag}: trained in {minutes:.1f} minutes")

eval_env = DroneNavEnv(size=SIZE, max_steps=MAX_STEPS, include_last_move=True, vision=vision)
rows = []
for steps in range(EVERY, TOTAL_STEPS + 1, EVERY):
    checkpoint = DQN.load(checkpoints / f"dqn_{steps}_steps")
    s = summarise([fly_dqn(eval_env, checkpoint, m) for m in split_seeds("scale_validation")])
    rows.append({"vision": vision, "seed": seed, "train_moves": steps, "n": s["n"],
                 "success_%": round(100 * s["success_rate"], 1), "crash_%": round(100 * s["collisions"] / s["n"], 1),
                 "timeout_%": round(100 * s["timeouts"] / s["n"], 1), "train_minutes": round(minutes, 1)})
    print(rows[-1])

with open(f"results/06_scale16/scale_curve_{vision}_seed{seed}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
