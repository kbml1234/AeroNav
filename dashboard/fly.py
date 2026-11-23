"""Flies one drone (saved model or A*) on one map and records every step, for the dashboard replay."""
from functools import lru_cache
from pathlib import Path

from stable_baselines3 import DQN

from src.astar_agent import next_action
from src.drone_env import DroneNavEnv

MODELS = Path(__file__).resolve().parent.parent / "results" / "models"

# model=None means A*
DRONES = {
    "A* (sees the whole map)": dict(model=None, size=8, vision="plus", wind_inputs=False),
    "Calm DQN (main study)": dict(model="final/dqn_calm_last_move_500k.zip", size=8, vision="plus", wind_inputs=False),
    "Gust DQN (storm twist)": dict(model="curve_gust/dqn_500000_steps.zip", size=8, vision="plus", wind_inputs=False),
    "A2 diamond + wind inputs": dict(model="wind_aware_diamond_seed0/dqn_500000_steps.zip", size=8, vision="diamond", wind_inputs=True),
    "A* on 16 × 16": dict(model=None, size=16, vision="plus", wind_inputs=False),
    "16 × 16, old reward (probe 1)": dict(model="scale16_diamond_seed0/dqn_5000000_steps.zip", size=16, vision="diamond", wind_inputs=False),
    "16 × 16, reward fix (probe 2)": dict(model="reward16_diamond_seed0/dqn_5000000_steps.zip", size=16, vision="diamond", wind_inputs=False),
}
DIRECTIONS = ["North", "East", "South", "West"]


@lru_cache(maxsize=None)
def load_model(path):
    return DQN.load(MODELS / path)


def fly(drone, map_seed, gust_pct=0, wind_dir=None, wind_pct=0):
    d = DRONES[drone]
    wind = (wind_dir, wind_pct / 100) if wind_dir is not None and wind_pct > 0 else None
    if d["wind_inputs"] and wind is None:
        wind = (0, 0.0)                       # a wind-aware drone always needs a wind (0% = calm)
    env = DroneNavEnv(size=d["size"], max_steps=200 if d["size"] == 8 else 400, include_last_move=True,
                      gust_prob=gust_pct / 100, wind=wind, include_wind=d["wind_inputs"], vision=d["vision"])
    obs, _ = env.reset(options={"map_seed": map_seed})
    model = load_model(d["model"]) if d["model"] else None
    steps = [{"pos": env.pos, "obs": obs.tolist(), "requested": None, "executed": None, "event": "start"}]
    while True:
        if model is None:
            action = next_action(env.grid, env.pos, env.goal)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        event = info["event"] if terminated else ("timeout" if truncated else "move")
        if info["wind"] is not None and event in ("move", "collision"):
            event = "wind " + info["wind"] if event == "move" else "wind crash"
        elif info["executed_action"] != info["requested_action"] and event in ("move", "collision"):
            event = "gust" if event == "move" else "gust crash"
        steps.append({"pos": env.pos, "obs": obs.tolist(), "requested": info["requested_action"],
                      "executed": info["executed_action"], "event": event, "reward": reward})
        if terminated or truncated:
            ending = info["event"] if terminated else "timeout"
            return {"grid": env.grid.copy(), "start": env.start, "goal": env.goal, "steps": steps,
                    "ending": ending, "size": d["size"], "vision": d["vision"], "wind_inputs": d["wind_inputs"]}


def input_labels(vision, wind_inputs):
    from src.drone_env import VISION
    names = {(-1, 0): "N", (0, 1): "E", (1, 0): "S", (0, -1): "W", (-2, 0): "NN", (0, 2): "EE", (2, 0): "SS",
             (0, -2): "WW", (-1, 1): "NE", (1, 1): "SE", (1, -1): "SW", (-1, -1): "NW"}
    labels = ["my row", "my column", "goal row", "goal column"]
    labels += [f"wall {names[o]}" for o in VISION[vision]]
    labels += ["last move N", "last move E", "last move S", "last move W"]
    if wind_inputs:
        labels += ["wind pushes N", "wind pushes E", "wind pushes S", "wind pushes W", "wind strength"]
    return labels
