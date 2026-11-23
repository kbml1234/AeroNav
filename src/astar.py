import heapq

from src.grid_utils import manhattan


def astar(grid, start, goal):
    rows, cols = grid.shape
    to_check = [(manhattan(start, goal), 0, start)]   # (score, steps so far, square)
    came_from = {start: None}
    best_steps = {start: 0}

    while to_check:
        score, steps, square = heapq.heappop(to_check)
        if square == goal:
            return build_route(came_from, goal)
        if steps > best_steps[square]:
            continue   # stale entry: a shorter route here was already found

        r, c = square
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            inside = 0 <= nr < rows and 0 <= nc < cols
            if not inside or grid[nr, nc] == 1:
                continue
            new_steps = steps + 1
            if (nr, nc) not in best_steps or new_steps < best_steps[(nr, nc)]:
                best_steps[(nr, nc)] = new_steps
                came_from[(nr, nc)] = square
                heapq.heappush(to_check, (new_steps + manhattan((nr, nc), goal), new_steps, (nr, nc)))

    return None


def build_route(came_from, goal):
    # walk backwards from the goal to the start, then flip the list
    route = [goal]
    while came_from[route[-1]] is not None:
        route.append(came_from[route[-1]])
    return route[::-1]
