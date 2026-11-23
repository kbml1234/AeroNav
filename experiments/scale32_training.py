"""Scale-up: train one 32 x 32 drone, D (direct) or K (curriculum: 8 x 8, then 16 x 16, then 32 x 32).

Run:  python -m experiments.scale32_training <D|K> <seed>
"""
import csv
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dqn_agent import make_dqn
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

torch.set_num_threads(1)   # one CPU thread per run
EVERY = 1_000_000
STAGES = {"D": [(32, 6_000_000)], "K": [(8, 1_000_000), (16, 2_000_000), (32, 3_000_000)]}
kind, SEED = sys.argv[1], int(sys.argv[2])
TAG = f"scale32_{kind}_seed{SEED}"
CHECKPOINTS = Path(f"results/models/{TAG}")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
train = load_splits()["train"]


def make_env(size):
    return DroneNavEnv(size=size, max_steps=25 * size, seed_range=(train["low"], train["high"]),
                       include_last_move=True, vision="diamond", shaped_reward=True, random_start_goal=True)


model = None
for i, (size, moves) in enumerate(STAGES[kind]):
    env = make_env(size)
    env.reset(seed=SEED + 1000 * i)
    if model is None:
        model = make_dqn(env, seed=SEED)
        # SB3 restarts its exploration schedule on every learn(), so count total moves instead
        model.exploration_schedule = lambda _: max(0.05, 1.0 - 0.95 * model.num_timesteps / 1_800_000)
    else:
        model.set_env(env)
    # keep counting moves across stages
    model.learn(total_timesteps=moves, reset_num_timesteps=(i == 0),
                callback=CheckpointCallback(save_freq=EVERY, save_path=str(CHECKPOINTS), name_prefix="dqn"))
    print(f"stage {i + 1}: {moves:,} moves on {size} x {size} done, total {model.num_timesteps:,}, exploration {model.exploration_rate:.3f}", flush=True)

eval_env = DroneNavEnv(size=32, max_steps=800, include_last_move=True, vision="diamond", random_start_goal=True)
model.save(CHECKPOINTS / "final_6000000")
rows = []
for f in sorted(CHECKPOINTS.glob("dqn_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1])):
    s = summarise([fly_dqn(eval_env, DQN.load(f), seed) for seed in split_seeds("scale32_validation")])
    rows.append({"checkpoint": f.stem, "success_%": 100 * s["success_rate"], "crash_%": 100 * s["collisions"] / s["n"],
                 "timeout_%": 100 * s["timeouts"] / s["n"]})
    print(rows[-1], flush=True)
with open(f"results/11_scale32/{TAG}_curve.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
