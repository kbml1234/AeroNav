from collections import deque

import numpy as np

from src.astar import astar
from src.map_generator import generate_map

small_grid = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
])


def bfs_moves(grid, start, goal):
    rows, cols = grid.shape
    moves = {start: 0}
    to_visit = deque([start])
    while to_visit:
        r, c = to_visit.popleft()
        if (r, c) == goal:
            return moves[(r, c)]
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in moves:
                moves[(nr, nc)] = moves[(r, c)] + 1
                to_visit.append((nr, nc))
    return None


def is_valid_route(grid, route, start, goal):
    if route[0] != start or route[-1] != goal:
        return False
    for (r1, c1), (r2, c2) in zip(route, route[1:]):
        if grid[r2, c2] == 1 or abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    return True


def test_route_on_small_map():
    route = astar(small_grid, (0, 0), (3, 3))
    assert is_valid_route(small_grid, route, (0, 0), (3, 3))
    assert len(route) - 1 == 6          # 6 moves, the shortest possible here


def test_no_route_gives_none():
    blocked = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    assert astar(blocked, (0, 0), (2, 2)) is None


def test_start_is_goal():
    assert astar(small_grid, (2, 1), (2, 1)) == [(2, 1)]


def test_matches_bfs_on_many_maps():
    for seed in range(200):
        size = 4 + seed % 5
        grid, start, goal = generate_map(size, 0.3, seed)
        route = astar(grid, start, goal)
        assert is_valid_route(grid, route, start, goal)
        assert len(route) - 1 == bfs_moves(grid, start, goal)
