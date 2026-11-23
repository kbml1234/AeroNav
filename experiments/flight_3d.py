"""3D view of recorded flights (display only: the simulation is still a flat 2D grid).

Run:  python -m experiments.flight_3d
"""
import json
from pathlib import Path

from stable_baselines3 import DQN

from src.astar_agent import next_action
from src.drone_env import DroneNavEnv

CALM = "results/models/final/dqn_calm_last_move_500k.zip"
GUST = "results/models/curve_gust/dqn_500000_steps.zip"
R2 = "results/models/startgoal_fix_seed2/dqn_500000_steps.zip"

FLIGHTS = [
    ("Map 30,050, calm air: A* (sees the whole map)", None, 30050, {}),
    ("Map 30,050, calm air: calm DQN (loops near a dead end)", CALM, 30050, {}),
    ("Map 30,000, 10% gusts: gust DQN (reaches the goal)", GUST, 30000, {"gust_prob": 0.1}),
    ("Map 30,000, 10% gusts: calm DQN (a gust pushes it off the map)", CALM, 30000, {"gust_prob": 0.1}),
    ("Map 70,109, random start/goal: random-trained DQN R", R2, 70109, {"random_start_goal": True}),
    ("Map 70,109, random start/goal: corner-trained DQN C (loops)", CALM, 70109, {"random_start_goal": True}),
]


def record(model_path, map_seed, settings):
    env = DroneNavEnv(include_last_move=True, **settings)
    obs, _ = env.reset(options={"map_seed": map_seed})
    model = DQN.load(model_path) if model_path else None
    steps = [{"pos": list(env.pos), "event": "start"}]
    while True:
        action = next_action(env.grid, env.pos, env.goal) if model is None else int(model.predict(obs, deterministic=True)[0])
        obs, _, terminated, truncated, info = env.step(action)
        event = info["event"] if terminated else ("timeout" if truncated else "move")
        if info["executed_action"] != info["requested_action"]:
            event = "gust crash" if event == "collision" else "gust"
        # on a crash, record where it tried to go
        tried = [env.pos[0] + [-1, 0, 1, 0][info["executed_action"]], env.pos[1] + [0, 1, 0, -1][info["executed_action"]]]
        steps.append({"pos": list(env.pos), "event": event, "tried": tried})
        if terminated or truncated:
            return {"grid": env.grid.tolist(), "start": list(env.start), "goal": list(env.goal), "steps": steps,
                    "ending": info["event"] if terminated else "timeout"}


flights = [{"label": label, **record(m, seed, s)} for label, m, seed, s in FLIGHTS]
for f in flights:
    print(f'{f["label"]}: {f["ending"]} after {len(f["steps"]) - 1} moves')

template = Path("dashboard/flight_3d_template.html").read_text()
Path("dashboard/flight_3d.html").write_text(template.replace("/*FLIGHTS*/[]", json.dumps(flights)))
print("saved dashboard/flight_3d.html")
