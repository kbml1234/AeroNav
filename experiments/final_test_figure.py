"""Chart of the one-time test results (reads results/03_main_test/final_test_summary.csv, flies nothing).

Run:  python -m experiments.final_test_figure
"""
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/03_main_test/final_test_summary.csv")))
gusts = [0, 5, 10, 20]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for agent, colour, shift in [("A*", "black", -0.6), ("calm DQN", "tab:gray", 0.0), ("gust DQN", "tab:blue", 0.6)]:
    means, lows, highs = [], [], []
    for g in gusts:
        v = [float(r["success_%"]) for r in rows if r["agent"] == agent and int(r["gust_%"]) == g]
        means.append(statistics.mean(v)); lows.append(min(v)); highs.append(max(v))
    x = [g + shift for g in gusts]
    label = "A* (sees the whole map, reference)" if agent == "A*" else f"{agent} (mean of 3 seeds, bar = range)"
    ax.errorbar(x, means, yerr=[[m - l for m, l in zip(means, lows)], [h - m for m, h in zip(means, highs)]],
                marker="o", color=colour, capsize=4, label=label)
ax.set_xticks(gusts)
ax.set_xlabel("gust strength (% of moves changed)")
ax.set_ylabel("reached the goal (% of 300 test maps)")
ax.set_ylim(0, 105)
ax.legend()
ax.set_title("Final test: 300 unseen maps, same gusts for every agent")
fig.tight_layout()
fig.savefig("results/figures/final_test_success.png", dpi=120)
print("saved results/figures/final_test_success.png")
