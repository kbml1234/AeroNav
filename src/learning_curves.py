import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling_mean(values, window):
    result = []
    for i in range(len(values)):
        recent = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        result.append(sum(recent) / len(recent) if recent else None)
    return result


def plot_learning_curves(history, shortest, filename, window=50):
    episodes = [row["episode"] for row in history]
    rewards = [row["reward"] for row in history]
    # steps only count on rounds that reached the goal: a crash after 1 move is not a "short route"
    goal_steps = [row["steps"] if row["ending"] == "goal" else None for row in history]
    success = [1 if row["ending"] == "goal" else 0 for row in history]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax1.plot(episodes, rewards, color="lightgray", linewidth=0.5, label="each round")
    ax1.plot(episodes, rolling_mean(rewards, window), color="tab:blue", label=f"average of last {window}")
    ax1.set_ylabel("total reward")
    ax1.legend(loc="lower right")

    goal_x = [e for e, s in zip(episodes, goal_steps) if s is not None]
    goal_y = [s for s in goal_steps if s is not None]
    ax2.scatter(goal_x, goal_y, color="lightgray", s=4, label="each round that reached the goal")
    ax2.plot(episodes, [v if v is not None else float("nan") for v in rolling_mean(goal_steps, window)],
             color="tab:green", label=f"average of last {window} (goal rounds only)")
    ax2.axhline(shortest, color="black", linestyle="--", linewidth=1, label=f"A* shortest = {shortest}")
    ax2.set_ylabel("steps to the goal")
    ax2.legend(loc="upper right")

    ax3.plot(episodes, [100 * v for v in rolling_mean(success, window)], color="tab:orange")
    ax3.set_ylabel(f"success rate %\n(last {window} rounds)")
    ax3.set_ylim(0, 100)
    ax3.set_xlabel("training round")

    fig.suptitle("Tabular Q-learning on one fixed 8x8 map (map seed 42)")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
