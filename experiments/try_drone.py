"""Fly one drone on one map and print the flight as a grid, to see where the model goes.

python -m experiments.try_drone <drone> <map_seed> [random]
"""
import sys
from collections import Counter

from stable_baselines3 import DQN

from src.astar_agent import next_action
from src.drone_env import DroneNavEnv

MODELS = {
    "C": "results/models/final/dqn_calm_last_move_500k.zip",
    **{f"R{s}": f"results/models/startgoal_fix_seed{s}/dqn_500000_steps.zip" for s in range(3)},
}
MOVES = "NESW"

drone, map_seed = sys.argv[1], int(sys.argv[2])
random_trip = len(sys.argv) > 3 and sys.argv[3] == "random"
env = DroneNavEnv(include_last_move=True, random_start_goal=random_trip)
obs, _ = env.reset(options={"map_seed": map_seed})
model = None if drone == "astar" else DQN.load(MODELS[drone])

visits = Counter([env.pos])
moves = []
while True:
    if model is None:
        action = next_action(env.grid, env.pos, env.goal)
    else:
        action = int(model.predict(obs, deterministic=True)[0])
    obs, reward, terminated, truncated, info = env.step(action)
    moves.append(MOVES[action])
    visits[env.pos] += 1
    if terminated or truncated:
        ending = info["event"] if terminated else "timeout"
        break

print(f"drone {drone}, map {map_seed}, start {env.start}, goal {env.goal}: {ending} after {len(moves)} moves")
for r in range(env.size):
    row = []
    for c in range(env.size):
        if env.grid[r, c] == 1:
            row.append(" #")
        elif (r, c) == env.start:
            row.append(" S")
        elif (r, c) == env.goal:
            row.append(" G")
        elif visits[(r, c)]:
            row.append(f"{min(visits[(r, c)], 99):2d}")
        else:
            row.append(" .")
    print(" ".join(row))
print("first 30 moves:", "".join(moves[:30]))
