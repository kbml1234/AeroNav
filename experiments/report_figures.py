"""The four figures for the research paper, made from the saved test flights and learning curves.

Run:  python -m experiments.report_figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dashboard.data import load_flights
from dashboard.draw import draw_flight
from dashboard.fly import fly
from src.stats import wilson_interval

OUT = "results/figures/report_"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
data = load_flights()


def rate_ci(g):
    k, n = int(g["success"].sum()), len(g)
    lo, hi = wilson_interval(k, n)
    p = k / n
    return 100 * p, 100 * (p - lo), 100 * (hi - p)


fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
for ax, map_seed, title in [(axes[0], 30000, "(a) map 30,000: goal in 14 moves"),
                            (axes[1], 30050, "(b) map 30,050: loop near a dead end, timeout")]:
    f = fly("Calm DQN (main study)", map_seed)
    small = draw_flight(f, show_vision=False)
    small.set_dpi(220)
    small.canvas.draw()
    ax.imshow(np.asarray(small.canvas.buffer_rgba()))
    plt.close(small)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT + "fig1_routes.png", dpi=200)
plt.close(fig)

main = data[data["experiment"] == "Main test: random gusts (8 × 8)"]
fig, ax = plt.subplots(figsize=(6, 3.8))
for drone, style in [("A*", "o-"), ("calm DQN", "s-"), ("gust DQN", "^-")]:
    xs, ys, err = [], [], [[], []]
    for p in [0, 5, 10, 20]:
        g = main[(main["drone"] == drone) & (main["strength"] == p)]
        y, lo, hi = rate_ci(g)
        xs.append(p); ys.append(y); err[0].append(lo); err[1].append(hi)
    ax.errorbar(xs, ys, yerr=err, fmt=style, capsize=3, label=drone, markersize=5)
ax.set_xlabel("gust probability per move (%)")
ax.set_ylabel("success rate (%)")
ax.set_xticks([0, 5, 10, 20])
ax.set_ylim(0, 102)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT + "fig2_main_test.png", dpi=200)
plt.close(fig)

vis = data[data["experiment"] == "Diamond vision (8 × 8)"]
fig, ax = plt.subplots(figsize=(6, 3.8))
for drone, label, style in [("D A*", "A* (full map)", "o-"), ("A2 diamond+wind", "diamond + wind inputs", "s-"),
                            ("B2 diamond", "diamond, no wind inputs", "s--"), ("A wind-aware", "plus + wind inputs", "^-"),
                            ("B wind-blind", "plus, no wind inputs", "^--")]:
    xs, ys, err = [], [], [[], []]
    for p in [0, 10, 20, 30]:
        g = vis[(vis["drone"] == drone) & (vis["strength"] == p)]
        y, lo, hi = rate_ci(g)
        xs.append(p); ys.append(y); err[0].append(lo); err[1].append(hi)
    ax.errorbar(xs, ys, yerr=err, fmt=style, capsize=3, label=label, markersize=5)
ax.set_xlabel("wind strength (%)")
ax.set_ylabel("success rate (%)")
ax.set_xticks([0, 10, 20, 30])
ax.set_ylim(0, 102)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "fig3_vision.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, prefix, folder, title in [(axes[0], "scale", "06_scale16", "(a) original reward"),
                                  (axes[1], "reward", "07_reward16", "(b) reward fix")]:
    for s in range(3):
        c = pd.read_csv(f"results/{folder}/{prefix}_curve_diamond_seed{s}.csv")
        ax.plot(c["train_moves"] / 1e6, c["success_%"], marker="o", markersize=3, label=f"seed {s}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training moves (millions)")
    ax.set_ylim(-2, 100)
axes[0].set_ylabel("validation success (%)")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "fig4_scale_curves.png", dpi=200)
plt.close(fig)
# Figure 5: 32 x 32 validation success during training, direct (D) vs curriculum (K), 3 seeds each
fig, ax = plt.subplots(figsize=(6, 3.8))
for kind, color, label in [("D", "tab:red", "D direct"), ("K", "tab:green", "K curriculum")]:
    for s in range(3):
        c = pd.read_csv(f"results/11_scale32/scale32_{kind}_seed{s}_curve.csv")
        moves = [int(x.split("_")[1]) / 1e6 for x in c["checkpoint"]]
        ax.plot(moves, c["success_%"], color=color, marker="o", markersize=3, label=f"{label}, seed {s}",
                linestyle="-" if s == 0 else ("--" if s == 1 else ":"))
ax.set_xlabel("training moves (millions)")
ax.set_ylabel("validation success (%)")
ax.set_ylim(0, 60)
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT + "fig5_scale32_curves.png", dpi=200)
plt.close(fig)
# Figure 6: 32 x 32 validation success during the 24M-move curriculum, with (M) and without (K) visit memory
fig, ax = plt.subplots(figsize=(6, 3.8))
for tag, folder, color, label in [("scale32long_K", "12_scale32_long", "tab:green", "K no memory"),
                                  ("scale32memory", "14_memory", "tab:blue", "M memory")]:
    for s in range(3):
        c = pd.read_csv(f"results/{folder}/{tag}_seed{s}_curve.csv")
        ax.plot(c["train_moves"] / 1e6, c["success_%"], color=color, marker="o", markersize=3, label=f"{label}, seed {s}",
                linestyle="-" if s == 0 else ("--" if s == 1 else ":"))
for x in (4, 12):
    ax.axvline(x, color="grey", linewidth=0.8)
ax.text(2, 97, "8 × 8", ha="center", fontsize=8)
ax.text(8, 97, "16 × 16", ha="center", fontsize=8)
ax.text(18, 97, "32 × 32", ha="center", fontsize=8)
ax.set_xlabel("training moves (millions)")
ax.set_ylabel("validation success on 32 × 32 (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(OUT + "fig6_memory_curves.png", dpi=200)
plt.close(fig)
print("saved 6 figures")
