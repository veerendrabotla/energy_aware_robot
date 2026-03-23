import numpy as np
from config import *
from dataset.dataset_generator import extract_features

def run_ann_navigation(grid, terrain, start, goal, model, phase=2, max_steps=500):
    cy, cx = start
    gy, gx = goal
    
    dy_goal = gy - cy
    dx_goal = gx - cx
    if abs(dy_goal) > abs(dx_goal):
        hd = SOUTH if dy_goal > 0 else NORTH
    else:
        hd = EAST if dx_goal > 0 else WEST
        
    path = [(cy, cx, hd)]
    status = "running"
    consecutive_spins = 0
    
    for _ in range(max_steps):
        if (cy, cx) == goal:
            status = "success"
            break
            
        features = extract_features(grid, terrain, (cy, cx, hd), goal, phase=phase)
        X = np.array(features).reshape(1, -1)
        action = model.predict(X)[0]
        
        if action == FORWARD:
            consecutive_spins = 0
            dy, dx = MOVES[hd]
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                status = "out_of_bounds"
                break
            if grid[ny, nx] == 1:
                status = "collision"
                break
            if phase == 2 and terrain is not None:
                slope = max(0, terrain[ny, nx] - terrain[cy, cx])
                if slope > MAX_SLOPE:
                    status = "collision_with_slope"
                    break
            cy, cx = ny, nx
        elif action == LEFT:
            hd = (hd - 1) % 4
            consecutive_spins += 1
        elif action == RIGHT:
            hd = (hd + 1) % 4
            consecutive_spins += 1
            
        if consecutive_spins >= 8:
            status = "spinning_failure"
            break
            
        path.append((cy, cx, hd))
        
    if status == "running":
        status = "max_steps"
        
    return path, status
