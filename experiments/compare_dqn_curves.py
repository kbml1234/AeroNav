"""Compare the validation curves of the DQN variants, and measure the loops left over.

Run:  python -m experiments.compare_dqn_curves
"""
import csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from src.drone_env import DroneNavEnv
from src.evaluate import split_seeds

VARIANTS = {"plan": "plan settings (8 inputs)",
            "revisit_penalty": "revisit penalty (8 inputs)",
            "last_move": "last move (12 inputs)"}
COLOURS = {"plan": "tab:gray", "revisit_penalty": "tab:blue", "last_move": "tab:green"}


def read_curve(name):
    with open(f"results/02_dqn_tuning/dqn_curve_{name}.csv") as f:
        return [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(f)]


def loop_length(positions, window=60):
    # the smallest p where the last `window` positions repeat every p moves (0 = no loop found)
    tail = positions[-window:]
    for p in range(1, window // 2 + 1):
        if all(tail[i] == tail[i + p] for i in range(len(tail) - p)):
            return p
    return 0


def fly_and_record(env, model, map_seed):
    obs, info = env.reset(options={"map_seed": map_seed})
    positions = [env.pos]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        positions.append(env.pos)
        if terminated or truncated:
            return ("timeout" if truncated else info["event"]), positions


fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for name, label in VARIANTS.items():
    rows = read_curve(name)
    x = [r["train_moves"] for r in rows]
    last3 = sum(r["success_%"] for r in rows[-3:]) / 3
    top.plot(x, [r["success_%"] for r in rows], marker="o", color=COLOURS[name],
             label=f"{label}: last-3 avg {last3:.1f}%")
    bottom.plot(x, [r["timeout_%"] for r in rows], marker="o", color=COLOURS[name], label=label)
top.set_ylabel("reached the goal (%)")
top.set_ylim(0, 100)
top.legend(loc="lower right")
top.set_title("DQN variants on the 300 validation maps (seed 0)")
bottom.set_ylabel("timeout / looping (%)")
bottom.set_ylim(0, 100)
bottom.set_xlabel("training moves")
fig.tight_layout()
fig.savefig("results/figures/day20_variants_compared.png", dpi=120)

rows = []
for name in VARIANTS:
    model = DQN.load(f"results/models/curve_{name}/dqn_500000_steps")
    env = DroneNavEnv(include_last_move=(name == "last_move"))
    lengths = Counter()
    for seed in split_seeds("validation"):
        ending, positions = fly_and_record(env, model, seed)
        if ending == "timeout":
            lengths[loop_length(positions)] += 1
    for p, count in sorted(lengths.items()):
        rows.append({"variant": name, "loop_moves": p, "maps": count})
    print(name, "timeouts:", sum(lengths.values()), "loop sizes (moves: maps):", dict(sorted(lengths.items())))

with open("results/02_dqn_tuning/day20_loop_lengths.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "loop_moves", "maps"])
    writer.writeheader()
    writer.writerows(rows)
