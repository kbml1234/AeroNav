"""Prototype freeze. Save the final calm DQN and everything needed to reproduce it.

Run:  python -m experiments.freeze_prototype
"""
import hashlib
import json
import shutil
from pathlib import Path

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import DQN

from src.dqn_agent import DQN_SETTINGS
from src.drone_env import DroneNavEnv
from src.evaluate import fly_dqn, load_splits, split_seeds, summarise
from src.render import record_episode, save_gif, save_png

SOURCE = Path("results/models/curve_last_move/dqn_500000_steps.zip")
FINAL = Path("results/models/final")
MODEL = FINAL / "dqn_calm_last_move_500k.zip"
FIGURES = Path("results/figures")

FINAL.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SOURCE, MODEL)
model = DQN.load(MODEL)
env = DroneNavEnv(include_last_move=True)

results = [fly_dqn(env, model, seed) for seed in split_seeds("validation")]
s = summarise(results)
print("validation:", s)


def flight_actions(map_seed):
    obs, _ = env.reset(options={"map_seed": map_seed})
    actions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return actions


goal_seed = 20000
loop_seed = next(r["map_seed"] for r in results if r["ending"] == "timeout")
goal_run = record_episode(DroneNavEnv(include_last_move=True), flight_actions(goal_seed), options={"map_seed": goal_seed})
loop_run = record_episode(DroneNavEnv(include_last_move=True), flight_actions(loop_seed), options={"map_seed": loop_seed})
save_gif(goal_run, FIGURES / "day21_demo_goal.gif")
save_png(goal_run, FIGURES / "day21_demo_goal.png")
save_png(loop_run, FIGURES / "day21_demo_loop.png")
print(f"demo: map {goal_seed} -> {goal_run['ending']} in {goal_run['steps']} moves; "
      f"map {loop_seed} -> {loop_run['ending']} after {loop_run['steps']} moves")

card = {
    "name": "calm DQN with last-move inputs",
    "file": str(MODEL),
    "sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    "selection_rule": "last checkpoint (500,000 training moves), chosen before any test map was used",
    "environment": {"size": 8, "obstacle_density": 0.15, "max_steps": 200, "include_last_move": True,
                    "inputs": 12, "rewards": {"goal": 100, "crash": -30, "move": -1}, "gusts": 0.0},
    "training": {"algorithm": "DQN", "total_moves": 500_000, "seed": 0,
                 "map_seeds": load_splits()["train"], "settings": DQN_SETTINGS,
                 "network": "12 -> 64 -> 64 -> 4", "parameters": sum(p.numel() for p in model.policy.q_net.parameters())},
    "validation": {"maps": "seeds 20000-20299", "n": s["n"], "success_%": round(100 * s["success_rate"], 1),
                   "collisions": s["collisions"], "timeouts": s["timeouts"],
                   "mean_goal_steps": round(s["mean_goal_steps"], 2)},
    "test": "not run yet when this card was saved",
    "versions": {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
                 "gymnasium": gymnasium.__version__},
    "demo": {"goal": f"validation map {goal_seed}", "loop": f"validation map {loop_seed}"},
}
(FINAL / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
print("saved", FINAL / "model_card.json")
