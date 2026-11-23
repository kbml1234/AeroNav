"""Failure studies. Replay failed test flights move by move and measure what went wrong.

Run:  python -m experiments.failure_study
"""
import csv
from collections import Counter
from pathlib import Path

from stable_baselines3 import DQN

from src.astar_agent import next_action
from src.drone_env import DroneNavEnv
from src.render import record_episode, save_png

OUT = Path("results/figures/failures")
OUT.mkdir(parents=True, exist_ok=True)
AGENTS = {"calm DQN": DQN.load("results/models/final/dqn_calm_last_move_500k.zip"),
          "gust DQN": DQN.load("results/models/curve_gust/dqn_500000_steps.zip"),
          "A*": None}


def replay(agent, map_seed, gust):
    env = DroneNavEnv(include_last_move=agent != "A*", gust_prob=gust)
    obs, _ = env.reset(options={"map_seed": map_seed})
    moves, path = [], [env.pos]
    while True:
        if agent == "A*":
            action = next_action(env.grid, env.pos, env.goal)
        else:
            action = int(AGENTS[agent].predict(obs, deterministic=True)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        moves.append((action, info["executed_action"]))
        path.append(env.pos)
        if terminated or truncated:
            return env, ("timeout" if truncated else info["event"]), moves, path


def loop_length(path, window=60):
    tail = path[-window:]
    for p in range(1, window // 2 + 1):
        if all(tail[i] == tail[i + p] for i in range(len(tail) - p)):
            return p
    return 0


def dead_end(env, square):
    r, c = square
    return sum(env._blocked(r + dr, c + dc) for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]) == 3


flights = list(csv.DictReader(open("results/03_main_test/final_test_flights.csv")))
failed = [f for f in flights if f["ending"] != "goal" and f["train_seed"] in ("0", "")]
rows = []
for f in failed:
    gust = int(f["gust_%"]) / 100
    env, ending, moves, path = replay(f["agent"], int(f["map_seed"]), gust)
    assert ending == f["ending"], "replay must match the test run"
    row = {"agent": f["agent"], "gust_%": f["gust_%"], "map_seed": f["map_seed"], "ending": ending, "moves": len(moves)}
    if ending == "timeout":
        n = loop_length(path)
        row["cause"] = f"loop of {n} moves" + (" into a dead end" if any(dead_end(env, s) for s in path[-n:]) else "")
    else:
        row["cause"] = "gust pushed it into a wall" if moves[-1][0] != moves[-1][1] else "agent's own move"
    rows.append(row)

with open("results/03_main_test/failure_study.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

for agent in AGENTS:
    for g in ("0", "10"):
        causes = Counter(r["cause"] for r in rows if r["agent"] == agent and r["gust_%"] == g)
        print(f"{agent} at {g}% gusts: {sum(causes.values())} failures", dict(causes.most_common()))

examples = {}
for r in rows:
    if r["agent"] == "calm DQN" and r["gust_%"] in ("0", "10"):
        examples.setdefault(r["cause"], r)
for cause, r in examples.items():
    env, ending, moves, path = replay(r["agent"], int(r["map_seed"]), int(r["gust_%"]) / 100)
    episode = record_episode(DroneNavEnv(include_last_move=True, gust_prob=int(r["gust_%"]) / 100),
                             [a for a, _ in moves], options={"map_seed": int(r["map_seed"])})
    name = f"calm_dqn_gust{r['gust_%']}_map{r['map_seed']}.png"
    save_png(episode, OUT / name)
    print("example:", cause, "->", name)
