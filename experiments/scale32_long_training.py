"""32 x 32, 4x longer training: train one drone, D (direct) or K (curriculum).

Run:  python -m experiments.scale32_long_training <D|K> <seed>
"""
import csv
import sys
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback

from src.dqn_agent import make_dqn
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

torch.set_num_threads(1)   # one CPU thread per run
EVERY = 2_000_000
EXPLORE = 7_200_000        # 30% of 24,000,000
STAGES = {"D": [(32, 24_000_000)], "K": [(8, 4_000_000), (16, 8_000_000), (32, 12_000_000)]}
kind, SEED = sys.argv[1], int(sys.argv[2])
TAG = f"scale32long_{kind}_seed{SEED}"
CHECKPOINTS = Path(f"results/models/{TAG}")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
OUT = Path("results/12_scale32_long")
OUT.mkdir(exist_ok=True)
CURVE = OUT / f"{TAG}_curve.csv"
train = load_splits()["train"]
eval_env = DroneNavEnv(size=32, max_steps=800, include_last_move=True, vision="diamond", random_start_goal=True)


def make_env(size):
    return DroneNavEnv(size=size, max_steps=25 * size, seed_range=(train["low"], train["high"]),
                       include_last_move=True, vision="diamond", shaped_reward=True, random_start_goal=True)


class SaveAndValidate(BaseCallback):
    # every 2M total moves: save the model and score it on the 32 x 32 validation maps, so progress shows while it runs
    def _on_step(self):
        if self.model.num_timesteps % EVERY == 0:
            path = CHECKPOINTS / f"dqn_{self.model.num_timesteps}_steps.zip"
            self.model.save(path)
            s = summarise([fly_dqn(eval_env, self.model, seed) for seed in split_seeds("scale32_validation")])
            row = {"train_moves": self.model.num_timesteps, "success_%": round(100 * s["success_rate"], 1),
                   "crash_%": round(100 * s["collisions"] / s["n"], 1), "timeout_%": round(100 * s["timeouts"] / s["n"], 1)}
            new = not CURVE.exists()
            with open(CURVE, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(row, flush=True)
        return True


if CURVE.exists():
    sys.exit(f"{CURVE} already exists; this run was already started.")
model = None
for i, (size, moves) in enumerate(STAGES[kind]):
    env = make_env(size)
    env.reset(seed=SEED + 1000 * i)
    if model is None:
        model = make_dqn(env, seed=SEED)
        model.exploration_schedule = lambda _: max(0.05, 1.0 - 0.95 * model.num_timesteps / EXPLORE)
    else:
        model.set_env(env)
    model.learn(total_timesteps=moves, reset_num_timesteps=(i == 0), callback=SaveAndValidate())
    print(f"stage {i + 1}: {moves:,} moves on {size} x {size} done, total {model.num_timesteps:,}", flush=True)
