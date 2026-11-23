"""32 x 32 fine-tuning with learning rate 0.0001.

Run:  python -m experiments.scale32_finetune <seed>
"""
import csv
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

torch.set_num_threads(1)
LR = 1e-4
EXTRA = 8_000_000
EVERY = 1_000_000
START = {0: 20_000_000, 1: 22_000_000, 2: 14_000_000}   # chosen long-run checkpoints
SEED = int(sys.argv[1])
TAG = f"scale32finetune_seed{SEED}"
CHECKPOINTS = Path(f"results/models/{TAG}")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
OUT = Path("results/13_scale32_finetune")
OUT.mkdir(exist_ok=True)
CURVE = OUT / f"{TAG}_curve.csv"
if CURVE.exists():
    sys.exit(f"{CURVE} already exists; this run was already started.")

train = load_splits()["train"]
env = DroneNavEnv(size=32, max_steps=800, seed_range=(train["low"], train["high"]), include_last_move=True,
                  vision="diamond", shaped_reward=True, random_start_goal=True)
env.reset(seed=SEED + 5000)
eval_env = DroneNavEnv(size=32, max_steps=800, include_last_move=True, vision="diamond", random_start_goal=True)
start_path = f"results/models/scale32long_K_seed{SEED}/dqn_{START[SEED]}_steps.zip"
model = DQN.load(start_path, env=env, custom_objects={"learning_rate": LR, "lr_schedule": lambda _: LR})
model.exploration_schedule = lambda _: 0.05


class SaveAndValidate(BaseCallback):
    def _on_step(self):
        done = self.model.num_timesteps - START[SEED]
        if done > 0 and done % EVERY == 0:
            path = CHECKPOINTS / f"dqn_plus{done}_steps.zip"
            self.model.save(path)
            s = summarise([fly_dqn(eval_env, self.model, seed) for seed in split_seeds("scale32_validation")])
            row = {"extra_moves": done, "success_%": round(100 * s["success_rate"], 1),
                   "crash_%": round(100 * s["collisions"] / s["n"], 1), "timeout_%": round(100 * s["timeouts"] / s["n"], 1),
                   "learning_rate": self.model.policy.optimizer.param_groups[0]["lr"]}
            new = not CURVE.exists()
            with open(CURVE, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(row, flush=True)
        return True


model.learn(total_timesteps=EXTRA, reset_num_timesteps=False, callback=SaveAndValidate())
print("done", model.num_timesteps, flush=True)
