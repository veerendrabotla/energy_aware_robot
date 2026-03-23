import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_2d_grid(grid, start, goal, astar_path=None, ann_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    cmap = ListedColormap(['white', 'black'])
    ax.imshow(grid, cmap=cmap, origin='upper')
    
    sy, sx = start
    gy, gx = goal
    ax.plot(sx, sy, 'go', markersize=10, label='Start')
    ax.plot(gx, gy, 'ro', markersize=10, label='Goal')
    
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    if astar_path:
        ay = [p[0] for p in astar_path]
        ax_x = [p[1] for p in astar_path]
        ax.plot(ax_x, ay, 'b-', linewidth=2, label='A* Path')
        
    if ann_path:
        ny = [p[0] for p in ann_path]
        nx = [p[1] for p in ann_path]
        ax.plot(nx, ny, 'g--', linewidth=2, label='ANN Path')
        
    ax.legend(loc='upper right')
    ax.set_title("2D Navigation Grid")
    
    return fig
