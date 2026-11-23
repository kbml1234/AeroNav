"""DQN tuning runs, one change at a time, judged on the validation maps only.

Run:  python -m experiments.dqn_validation_runs <variant>
"""
import csv
import sys
import time
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from src.drone_env import DroneNavEnv
from src.dqn_agent import make_dqn
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

SEED = 0
VARIANTS = {
    "first": (50_000, {}, "first run, default settings"),
    "less_exploring": (50_000, {"exploration_fraction": 0.1}, "decrease epsilon (decay over 10% instead of 30%)"),
    "longer": (200_000, {}, "the plan's next step: train 200,000 moves instead of 50,000"),
}

name = sys.argv[1]
steps, changes, why = VARIANTS[name]
Path("results/models").mkdir(parents=True, exist_ok=True)
train = load_splits()["train"]
monitor_file = f"results/02_dqn_tuning/dqn_{name}_monitor.csv"
env = Monitor(DroneNavEnv(seed_range=(train["low"], train["high"])), monitor_file)
env.reset(seed=SEED)
model = make_dqn(env, seed=SEED, **changes)

start = time.time()
model.learn(total_timesteps=steps)
seconds = time.time() - start
model.save(f"results/models/dqn_{name}")

with open(monitor_file) as f:
    rounds = list(csv.reader(f))[2:]
train_goals = sum(float(r[0]) > 0 for r in rounds)   # only a goal round can end with a positive total

eval_env = DroneNavEnv()
s = summarise([fly_dqn(eval_env, model, seed) for seed in split_seeds("validation")])
row = {"variant": name, "why": why, "train_moves": steps, "changes": str(changes), "seconds": round(seconds),
       "train_rounds": len(rounds), "train_goals": train_goals, "val_success_%": round(100 * s["success_rate"], 1),
       "val_crashes": s["collisions"], "val_timeouts": s["timeouts"], "val_mean_goal_steps": s["mean_goal_steps"]}

out = Path("results/02_dqn_tuning/dqn_validation_runs.csv")
new = not out.exists()
with open(out, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if new:
        writer.writeheader()
    writer.writerow(row)
print(row)
