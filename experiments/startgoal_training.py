"""Random start/goal follow-up: train one DQN R with random start and goal.

Run:  python -m experiments.startgoal_training <seed> [fix]
"""
import csv
import sys
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dqn_agent import make_dqn
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

TOTAL_STEPS = 500_000
EVERY = 50_000
SEED = int(sys.argv[1])
FIX = len(sys.argv) > 2 and sys.argv[2] == "fix"
TAG = f"startgoal_fix_seed{SEED}" if FIX else f"startgoal_seed{SEED}"
CHECKPOINTS = Path(f"results/models/{TAG}")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

train = load_splits()["train"]
env = DroneNavEnv(seed_range=(train["low"], train["high"]), include_last_move=True, random_start_goal=True,
                  shaped_reward=FIX)
env.reset(seed=SEED)
model = make_dqn(env, seed=SEED)
model.learn(total_timesteps=TOTAL_STEPS,
            callback=CheckpointCallback(save_freq=EVERY, save_path=str(CHECKPOINTS), name_prefix="dqn"))

eval_env = DroneNavEnv(include_last_move=True, random_start_goal=True)
rows = []
for steps in range(EVERY, TOTAL_STEPS + 1, EVERY):
    checkpoint = DQN.load(CHECKPOINTS / f"dqn_{steps}_steps")
    s = summarise([fly_dqn(eval_env, checkpoint, seed) for seed in split_seeds("startgoal_validation")])
    rows.append({"train_moves": steps, "success_%": 100 * s["success_rate"],
                 "crash_%": 100 * s["collisions"] / s["n"], "timeout_%": 100 * s["timeouts"] / s["n"],
                 "mean_goal_steps": s["mean_goal_steps"]})
    print(rows[-1], flush=True)

with open(f"results/10_startgoal/{TAG.replace('_seed', '_curve_seed')}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
