from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from src.drone_env import ACTIONS

ENDING_TEXT = {"goal": "reached goal", "collision": "crashed", "timeout": "timed out", None: "still flying"}


def record_episode(env, actions, seed=None, options=None):
    env.reset(seed=seed, options=options)
    path = [env.pos]
    ending = None
    crash_action = None

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if info["event"] == "collision":
            ending = "collision"
            crash_action = info["executed_action"]
        else:
            path.append(env.pos)
            if info["event"] == "goal":
                ending = "goal"
            elif truncated:
                ending = "timeout"
        if terminated or truncated:
            break

    return {
        "grid": env.grid, "start": env.start, "goal": env.goal, "map_seed": env.map_seed,
        "path": path, "ending": ending, "crash_action": crash_action, "steps": env.steps,
    }


def draw_slashed_clock(ax, c, r):
    ax.add_patch(Circle((c, r), 0.28, facecolor="white", edgecolor="darkorange", linewidth=2, zorder=4))
    ax.plot([c, c], [r, r - 0.18], color="darkorange", linewidth=2, zorder=5)
    ax.plot([c, c + 0.13], [r, r], color="darkorange", linewidth=2, zorder=5)
    ax.plot([c - 0.26, c + 0.26], [r + 0.26, r - 0.26], color="red", linewidth=2.5, zorder=6)


def draw_episode(ax, episode, upto=None):
    grid, start, goal = episode["grid"], episode["start"], episode["goal"]
    path = episode["path"] if upto is None else episode["path"][:upto]
    finished = upto is None or upto == len(episode["path"])
    ending = episode["ending"] if finished else None
    rows, cols = grid.shape
    ax.clear()

    for r in range(rows):
        for c in range(cols):
            color = "black" if grid[r, c] == 1 else "white"
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor="lightgray"))
    ax.add_patch(Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, facecolor="#9FE1CB", edgecolor="lightgray"))
    ax.add_patch(Rectangle((goal[1] - 0.5, goal[0] - 0.5), 1, 1, facecolor="#FAC775", edgecolor="lightgray"))
    ax.text(start[1], start[0], "S", ha="center", va="center", fontweight="bold")
    ax.text(goal[1], goal[0], "G", ha="center", va="center", fontweight="bold")

    ax.plot([c for r, c in path], [r for r, c in path], color="gray", linewidth=2, zorder=2)

    for (r, c), n in Counter(path).items():
        if n >= 2:
            ax.text(c + 0.47, r - 0.45, str(n), ha="right", va="top", fontsize=9, fontweight="bold",
                    color="dimgray", zorder=7)

    r, c = path[-1]
    if ending == "timeout":
        draw_slashed_clock(ax, c, r)
    else:
        ax.add_patch(Circle((c, r), 0.2, color="#378ADD", zorder=4))
    if ending == "collision":
        dr, dc = ACTIONS[episode["crash_action"]]
        ax.annotate("", xy=(c + 0.9 * dc, r + 0.9 * dr), xytext=(c, r), annotation_clip=False,
                    arrowprops=dict(color="red", width=2, headwidth=9), zorder=5)

    steps = episode["steps"] if finished else len(path) - 1
    ax.set_title(f"map seed {episode['map_seed']}  |  {steps} steps  |  {ENDING_TEXT[ending]}", fontsize=9)
    ax.set_xlim(-1, cols)
    ax.set_ylim(rows, -1)        # flipped so row 0 is at the top
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))


def save_png(episode, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_episode(ax, episode)
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_gif(episode, filename, fps=3):
    fig, ax = plt.subplots(figsize=(5, 5))
    frames = range(1, len(episode["path"]) + 1)
    animation = FuncAnimation(fig, lambda t: draw_episode(ax, episode, upto=t), frames=frames)
    animation.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
