import numpy as np
from src.map_generator import is_solvable, generate_map

def test_solvable_map():
    grid = np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    assert is_solvable(grid, (0, 0), (2, 3)) == True
def test_blocked_map():
    grid = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert is_solvable(grid, (0, 0), (2, 3)) == False
def test_same_seed_same_map():
    grid1, start1, goal1 = generate_map(8, 0.2, 42)
    grid2, start2, goal2 = generate_map(8, 0.2, 42)
    assert np.array_equal(grid1, grid2)

def test_generated_map_is_solvable():
    grid, start, goal = generate_map(8, 0.2, 42)
    assert is_solvable(grid, start, goal)


import pytest


def test_grid_is_size_by_size():
    for size in [4, 8, 12]:
        grid, start, goal = generate_map(size, 0.15, 1)
        assert grid.shape == (size, size)


def test_start_and_goal_fixed_and_open():
    for seed in range(50):
        grid, start, goal = generate_map(8, 0.3, seed)
        assert start == (0, 0) and goal == (7, 7)
        assert grid[start] == 0 and grid[goal] == 0


def test_wall_fraction_close_to_density():
    # average over many maps; a little under 0.15 is expected because
    # S and G are forced open and unsolvable (wall-heavy) maps get thrown away
    fractions = [generate_map(12, 0.15, seed)[0].mean() for seed in range(200)]
    assert 0.10 < sum(fractions) / len(fractions) < 0.16


def test_too_small_size_is_refused():
    with pytest.raises(ValueError):
        generate_map(3, 0.15, 1) #Must be careful as when using generate_map the function has been set to accept inputs greater than or equal to 4, So any input less will cause an error
    generate_map(4, 0.15, 1)   # 4 is the minimum, so this must work


def test_many_seeds_all_solvable():
    for seed in range(100):
        grid, start, goal = generate_map(8, 0.3, seed)
        assert is_solvable(grid, start, goal)


def test_different_seeds_give_different_maps():
    maps = {generate_map(8, 0.15, seed)[0].tobytes() for seed in range(20)}
    assert len(maps) == 20
