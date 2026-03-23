import numpy as np

def generate_2d_map(size=20, density=0.2):
    """Generates a 2D grid with 0=free space, 1=obstacle."""
    grid = np.random.choice([0, 1], size=(size, size), p=[1 - density, density])
    return grid

def get_valid_start_goal(grid):
    size = grid.shape[0]
    while True:
        start = (np.random.randint(size), np.random.randint(size))
        goal = (np.random.randint(size), np.random.randint(size))
        if grid[start] == 0 and grid[goal] == 0 and start != goal:
            return start, goal
