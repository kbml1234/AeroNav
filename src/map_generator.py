import numpy as np
from collections import deque

def is_solvable(grid, start, goal):
    rows, cols = grid.shape
    to_visit = deque([start])                 # step 1
    seen = {start}                            # step 1

    while to_visit:                           # step 5
        r, c = to_visit.popleft()             # step 2
        if (r, c) == goal:                    # step 3
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   # up, down, left, right
            nr, nc = r + dr, c + dc
            inside = 0 <= nr < rows and 0 <= nc < cols
            if inside and grid[nr, nc] == 0 and (nr, nc) not in seen:   # step 4
                seen.add((nr, nc))
                to_visit.append((nr, nc))
    return False                              # step 6


MIN_SIZE = 4   # smaller maps are trivial and waste RL training time


def generate_map(size, density, seed):
    if size < MIN_SIZE:
        raise ValueError(f"size must be at least {MIN_SIZE}, got {size}")
    rng = np.random.default_rng(seed)
    start = (0, 0)
    goal = (size - 1, size - 1)
    while True:
        grid = (rng.random((size, size)) < density).astype(int)
        grid[start] = 0
        grid[goal] = 0
        if is_solvable(grid, start, goal):
            return grid, start, goal

# start/goal route at least 70% of corner-to-corner
MIN_ROUTE_SHARE = 0.7


def min_route_length(size):
    corner_to_corner = 2 * (size - 1)
    return int(np.floor(MIN_ROUTE_SHARE * corner_to_corner + 0.5))   # always rounds .5 up


def route_lengths(grid, start):
    # BFS route length from start to every reachable square
    rows, cols = grid.shape
    lengths = {start: 0}
    to_visit = deque([start])
    while to_visit:
        r, c = to_visit.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in lengths:
                lengths[(nr, nc)] = lengths[(r, c)] + 1
                to_visit.append((nr, nc))
    return lengths


def generate_trip_map(size, density, seed, start_tries=20):
    if size < MIN_SIZE:
        raise ValueError(f"size must be at least {MIN_SIZE}, got {size}")
    rng = np.random.default_rng([seed, 7])   # a separate stream from generate_map's corner maps
    need = min_route_length(size)
    while True:
        grid = (rng.random((size, size)) < density).astype(int)
        open_squares = [tuple(int(v) for v in sq) for sq in np.argwhere(grid == 0)]
        for _ in range(start_tries):
            start = open_squares[rng.integers(len(open_squares))]
            far = sorted(sq for sq, moves in route_lengths(grid, start).items() if moves >= need)
            if far:
                goal = far[rng.integers(len(far))]
                return grid, start, goal
        # no start on this grid had a far enough goal: draw a new grid
