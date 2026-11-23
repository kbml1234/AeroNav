"""Demo: draw one episode of each ending on the seed-42 map.

Run:  python -m experiments.render_demo
"""
from pathlib import Path

from src.drone_env import DroneNavEnv
from src.render import record_episode, save_gif, save_png

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

goal_run = record_episode(DroneNavEnv(), [1] * 7 + [2] * 7, seed=42)
crash_run = record_episode(DroneNavEnv(), [1, 1, 2, 3], seed=42)
timeout_run = record_episode(DroneNavEnv(max_steps=10), [1, 3] * 10, seed=42)

for name, run in [("goal", goal_run), ("crash", crash_run), ("timeout", timeout_run)]:
    save_png(run, OUT / f"day6_{name}.png")
    print(f"{name}: ending={run['ending']}, steps={run['steps']}")

save_gif(goal_run, OUT / "day6_goal.gif")
save_gif(timeout_run, OUT / "day6_timeout.gif")
print("saved to", OUT)
