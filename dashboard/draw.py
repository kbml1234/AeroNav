"""Draws a flight (map, walls, start, goal, path, drone) for the dashboard replay. Plain matplotlib."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WALL, PATH, CRASH, PUSHED, SEEN = "#2F2F2F", "#1F5FA8", "#C0392B", "#E67E22", "#9BBBD9"


def draw_flight(flight, upto=None, show_vision=True):
    steps = flight["steps"] if upto is None else flight["steps"][:upto + 1]
    size = flight["size"]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=90)
    ax.imshow(flight["grid"], cmap=matplotlib.colors.ListedColormap(["white", WALL]), vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#BBBBBB", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(labelsize=7, length=0)

    if show_vision:
        from src.drone_env import VISION
        r0, c0 = steps[-1]["pos"]
        for dr, dc in VISION[flight["vision"]]:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < size and 0 <= c < size:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=SEEN, linewidth=1.5))

    sr, sc = flight["start"]
    gr, gc = flight["goal"]
    fs = 10 if size == 8 else 6
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=fs, weight="bold", color="#1E7B34")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=fs, weight="bold", color="#1E7B34")

    pts = np.array([[s["pos"][1], s["pos"][0]] for s in steps], dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], color=PATH, linewidth=2 if size == 8 else 1.4)
    pushed = [s for s in steps if s["event"] in ("gust", "wind replace", "wind drift")]
    if pushed:
        ax.plot([s["pos"][1] for s in pushed], [s["pos"][0] for s in pushed], "s", color=PUSHED,
                markersize=5 if size == 8 else 3, linestyle="none")

    last = steps[-1]
    r, c = last["pos"]
    ax.plot([c], [r], "o", color=PATH, markersize=9 if size == 8 else 5)
    if last["event"] in ("collision", "gust crash", "wind crash"):
        dc, dr = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[last["executed"]]
        ax.plot([c + 0.5 * dc], [r + 0.5 * dr], "x", color=CRASH, markersize=12 if size == 8 else 7, mew=2.5)
    fig.tight_layout()
    return fig


def draw_vision(vision):
    from src.drone_env import VISION
    img = np.zeros((5, 5))
    for dr, dc in VISION[vision]:
        img[dr + 2, dc + 2] = 1
    img[2, 2] = 2
    fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=110)
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap(["white", SEEN, PATH]), vmin=0, vmax=2)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="#BBBBBB", linewidth=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    fig.tight_layout()
    return fig
