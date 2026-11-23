from src.grid_utils import manhattan

def test_manhattan():
    assert manhattan((0, 1), (2, 3)) == 4
    assert manhattan((1,2), (2,3)) == 2
