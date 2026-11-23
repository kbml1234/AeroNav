"""Check the calm and gust DQNs on the validation maps at every gust strength.

Run:  python -m experiments.gust_validation_check
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, split_seeds, summarise

MODELS = {"calm DQN": "results/models/final/dqn_calm_last_move_500k.zip",
          "gust DQN": "results/models/curve_gust/dqn_500000_steps.zip"}
GUSTS = [0.0, 0.05, 0.10, 0.20]

rows = []
for name, path in MODELS.items():
    model = DQN.load(path)
    for p in GUSTS:
        env = DroneNavEnv(include_last_move=True, gust_prob=p)
        s = summarise([fly_dqn(env, model, seed) for seed in split_seeds("validation")])
        rows.append({"model": name, "gust_%": int(100 * p), "n": s["n"], "success_%": round(100 * s["success_rate"], 1),
                     "crashes": s["collisions"], "timeouts": s["timeouts"],
                     "mean_goal_steps": round(s["mean_goal_steps"], 2)})
        print(rows[-1])

with open("results/03_main_test/gust_validation_check.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, colour in [("calm DQN", "tab:gray"), ("gust DQN", "tab:blue")]:
    r = [row for row in rows if row["model"] == name]
    ax.plot([x["gust_%"] for x in r], [x["success_%"] for x in r], marker="o", color=colour, label=name)
    for x in r:
        ax.annotate(f"{x['success_%']}", (x["gust_%"], x["success_%"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=colour)
ax.set_xlabel("gust strength (% of moves changed)")
ax.set_ylabel("reached the goal (% of 300 validation maps)")
ax.set_xticks([0, 5, 10, 20])
ax.set_ylim(0, 100)
ax.legend()
ax.set_title("Validation check (not the test): one training seed each")
fig.tight_layout()
fig.savefig("results/figures/day24_gust_validation_check.png", dpi=120)
