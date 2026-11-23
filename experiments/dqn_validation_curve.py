"""How do the validation rates change as training goes on?

Run:  python -m experiments.dqn_validation_curve <variant>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from src.drone_env import DroneNavEnv
from src.dqn_agent import make_dqn
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise

TOTAL_STEPS = 500_000
EVERY = 50_000
SEED = 0
VARIANTS = {
    "plan": {},
    "revisit_penalty": {"revisit_penalty": 2.0},
    "last_move": {"include_last_move": True},
    "gust": {"include_last_move": True, "gust_prob": 0.10},
}
name = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
tag = name if SEED == 0 else f"{name}_seed{SEED}"
CHECKPOINTS = Path(f"results/models/curve_{tag}")
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

train = load_splits()["train"]
env = DroneNavEnv(seed_range=(train["low"], train["high"]), **VARIANTS[name])
env.reset(seed=SEED)
model = make_dqn(env, seed=SEED)
model.learn(total_timesteps=TOTAL_STEPS,
            callback=CheckpointCallback(save_freq=EVERY, save_path=str(CHECKPOINTS), name_prefix="dqn"))

# same inputs as in training, normal reward
eval_env = DroneNavEnv(include_last_move=VARIANTS[name].get("include_last_move", False))
rows = []
for steps in range(EVERY, TOTAL_STEPS + 1, EVERY):
    checkpoint = DQN.load(CHECKPOINTS / f"dqn_{steps}_steps")
    s = summarise([fly_dqn(eval_env, checkpoint, seed) for seed in split_seeds("validation")])
    rows.append({"train_moves": steps, "success_%": 100 * s["success_rate"],
                 "crash_%": 100 * s["collisions"] / s["n"], "timeout_%": 100 * s["timeouts"] / s["n"],
                 "mean_goal_steps": s["mean_goal_steps"]})
    gust = VARIANTS[name].get("gust_prob", 0.0)
    if gust:
        gust_env = DroneNavEnv(include_last_move=True, gust_prob=gust)
        g = summarise([fly_dqn(gust_env, checkpoint, seed) for seed in split_seeds("validation")])
        rows[-1].update({"gust_success_%": 100 * g["success_rate"], "gust_crash_%": 100 * g["collisions"] / g["n"],
                         "gust_timeout_%": 100 * g["timeouts"] / g["n"]})
    print(rows[-1])

with open(f"results/02_dqn_tuning/dqn_curve_{tag}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

x = [r["train_moves"] for r in rows]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, [r["success_%"] for r in rows], marker="o", color="tab:green", label="reached the goal")
ax.plot(x, [r["timeout_%"] for r in rows], marker="o", color="tab:orange", label="timeout (looping)")
ax.plot(x, [r["crash_%"] for r in rows], marker="o", color="tab:red", label="crash")
ax.set_xlabel("training moves")
ax.set_ylabel("% of 300 validation maps")
ax.set_ylim(0, 100)
ax.legend()
ax.set_title(f"DQN on unseen validation maps as training goes on ({name}, seed {SEED})")
fig.tight_layout()
fig.savefig(f"results/figures/dqn_curve_{tag}.png", dpi=120)
