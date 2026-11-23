import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.grid_utils import manhattan
from src.map_generator import generate_map, generate_trip_map

ACTIONS = {
    0: (-1, 0),   # up    (north): row - 1
    1: (0, 1),    # right (east):  col + 1
    2: (1, 0),    # down  (south): row + 1
    3: (0, -1),   # left  (west):  col - 1
}
def move(pos, action):
    dr, dc = ACTIONS[action]
    return (pos[0] + dr,pos[1] + dc)
def step_outcome(grid, pos, action, goal):
    new_pos = move(pos, action)
    r, c = new_pos
    rows, cols = grid.shape

    # check order: off the grid, goal, wall
    inside = 0 <= r < rows and 0 <= c < cols
    if not inside:
        return pos, -30, True, "collision"

    if new_pos == goal: #The position where the final ending point is
        return new_pos, 100, True, "goal"

    if grid[r, c] == 1:
        return pos, -30, True, "collision"

    return new_pos, -1, False, "move"


MAX_STEPS =200

# the squares each kind of vision checks, as (row change, col change) in the protocol's fixed order:
# N, E, S, W first (the same 4 flags as always), then NN, EE, SS, WW, then NE, SE, SW, NW
VISION = {
    "plus": [(-1, 0), (0, 1), (1, 0), (0, -1)],
    "diamond": [(-1, 0), (0, 1), (1, 0), (0, -1),
                (-2, 0), (0, 2), (2, 0), (0, -2),
                (-1, 1), (1, 1), (1, -1), (-1, -1)],
}
def run_episode(grid, start, goal, actions, max_steps=MAX_STEPS):
    pos = start
    steps = 0
    log = []

    for action in actions:
        executed = action
        new_pos, reward, terminated, event = step_outcome(grid, pos, executed, goal)
        steps = steps + 1
        truncated = steps >= max_steps and not terminated

        log.append({
            "step": steps,
            "requested_action": action,
            "executed_action": executed,
            "from": pos,
            "to": new_pos,
            "reward": reward,
            "event": event,
            "terminated": terminated,
            "truncated": truncated,
        })

        pos = new_pos
        if terminated or truncated:
            break

    return log


class DroneNavEnv(gym.Env):
    def __init__(self, size=8, obstacle_density=0.15, max_steps=MAX_STEPS, seed_range=(0, 2**31 - 1),
                 revisit_penalty=0.0, include_last_move=False, gust_prob=0.0,
                 wind=None, include_wind=False, drift_share=0.1, vision="plus", shaped_reward=False,
                 random_start_goal=False, visit_memory=False):
        super().__init__()
        self.size = size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        # map seeds are drawn from this range (scenarios/splits.json)
        self.seed_range = seed_range
        self.revisit_penalty = revisit_penalty
        self.include_last_move = include_last_move
        # chance that a gust swaps the move for one of the other 3
        self.gust_prob = gust_prob
        # wind: None, (direction, strength), or "schedule". When it acts, drift_share of the time it pushes
        # 1 extra square, otherwise it replaces the move
        if include_wind and wind is None:
            raise ValueError("include_wind needs a wind")
        self.wind = wind
        self.include_wind = include_wind
        self.drift_share = drift_share
        self.episodes = 0
        # plus: 4 neighbours, diamond: 12 squares within 2 steps
        if vision not in VISION:
            raise ValueError(f"vision must be one of {list(VISION)}")
        self.vision = vision
        # reward fix: crash = -(max_steps + 1), +10 / -10 for closer / farther
        self.shaped_reward = shaped_reward
        # random start/goal
        self.random_start_goal = random_start_goal
        # memory: one input per square in view, 1 if visited earlier this flight
        self.visit_memory = visit_memory

        self.action_space = spaces.Discrete(4)
        # obs: row, col, goal row, goal col, wall flags, then last move and wind if on; all between 0 and 1
        n_obs = 4 + len(VISION[vision]) + (4 if include_last_move else 0) + (5 if include_wind else 0)
        n_obs += len(VISION[vision]) if visit_memory else 0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.episodes = 0

        if options is not None and "map_seed" in options:
            self.map_seed = options["map_seed"]
        else:
            self.map_seed = int(self.np_random.integers(*self.seed_range))

        make_map = generate_trip_map if self.random_start_goal else generate_map
        self.grid, self.start, self.goal = make_map(self.size, self.obstacle_density, self.map_seed)
        self.pos = self.start
        self.steps = 0
        self.visited = {self.start}
        self.last_move = None
        # paired gusts: depend only on (map seed, strength), so every drone gets the same gusts
        self.gust_rng = np.random.default_rng([self.map_seed, round(self.gust_prob * 1000)])
        wind = options["wind"] if options is not None and "wind" in options else self.wind
        if wind == "schedule":
            # direction stays for 4 flights while strength goes 0, 10, 20, 30%
            i = self.episodes
            wind = ((i // 4) % 4, [0.0, 0.1, 0.2, 0.3][i % 4])
        self.episodes = self.episodes + 1
        self.wind_dir, self.wind_strength = wind if wind is not None else (None, 0.0)
        # paired wind: the wind numbers depend only on (map seed, direction, strength)
        self.wind_rng = np.random.default_rng([self.map_seed, 100 + (self.wind_dir or 0), round(self.wind_strength * 1000)])
        return self._get_obs(), {"map_seed": self.map_seed, "wind": wind}

    def _wind_effect(self):
        # always draw both numbers, wind or not, so the wind never depends on what the drone does
        u = self.wind_rng.random()
        v = self.wind_rng.random()
        if self.wind_dir is None or u >= self.wind_strength:
            return None
        return "drift" if v < self.drift_share else "replace"

    def _apply_gust(self, action):
        # always draw both numbers, gust or not, so the gust schedule never depends on what the agent does
        u = self.gust_rng.random()
        k = int(self.gust_rng.integers(3))
        if u < self.gust_prob:
            others = [a for a in range(4) if a != action]
            return others[k]
        return action

    def step(self, action):
        action = int(action)
        distance_before = manhattan(self.pos, self.goal)
        executed = self._apply_gust(action)
        wind = self._wind_effect()
        if wind == "replace":
            executed = self.wind_dir
        self.pos, reward, terminated, event = step_outcome(self.grid, self.pos, executed, self.goal)
        self.steps = self.steps + 1
        # last move = the move that actually happened
        self.last_move = executed
        if wind == "drift" and not terminated:
            # drift: pushed 1 more square in the same step; where it ends decides crash or goal
            self.pos, reward, terminated, event = step_outcome(self.grid, self.pos, self.wind_dir, self.goal)
            self.last_move = self.wind_dir
        if self.shaped_reward:
            if event == "collision":
                reward = -(self.max_steps + 1)
            else:
                reward = reward + 10 * (distance_before - manhattan(self.pos, self.goal))
        if event == "move":
            if self.pos in self.visited:
                reward = reward - self.revisit_penalty
            self.visited.add(self.pos)
        truncated = self.steps >= self.max_steps and not terminated

        info = {
            "event": event,
            "requested_action": action,
            "executed_action": executed,
            "steps": self.steps,
            "wind": wind,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _blocked(self, r, c):
        # the edge of the map counts as an obstacle, same as a wall
        inside = 0 <= r < self.size and 0 <= c < self.size
        return not inside or self.grid[r, c] == 1

    def _get_obs(self):
        r, c = self.pos
        n = self.size - 1
        flags = [float(self._blocked(r + dr, c + dc)) for dr, dc in VISION[self.vision]]
        obs = [r / n, c / n, self.goal[0] / n, self.goal[1] / n] + flags
        if self.include_last_move:
            obs += [1.0 if self.last_move == a else 0.0 for a in range(4)]
        if self.include_wind:
            obs += [1.0 if self.wind_dir == a else 0.0 for a in range(4)] + [self.wind_strength]
        if self.visit_memory:
            obs += [float((r + dr, c + dc) in self.visited) for dr, dc in VISION[self.vision]]
        return np.array(obs, dtype=np.float32)
