import numpy as np # type: ignore
from config import NORTH, SOUTH, EAST, WEST, FORWARD, LEFT, RIGHT, MOVES, MAX_SLOPE, DISTANCE_COST, TURN_COST, SLOPE_COST # type: ignore
from dataset.dataset_generator import extract_features # type: ignore

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
    consecutive_blocked = 0  # Track how many times forward is blocked in a row
    
    for _ in range(max_steps):
        if (cy, cx) == goal:
            status = "success"
            break
            
        features = extract_features(grid, terrain, (cy, cx, hd), goal, phase=phase)
        X = np.array(features).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        
        # Hysteresis: bias keeping straight to minimize Turn Penalty energy costs
        HYSTERESIS_BIAS = 0.25
        probs[FORWARD] += HYSTERESIS_BIAS
        action = np.argmax(probs)
        
        if action == FORWARD:
            dy, dx = MOVES[hd]
            ny, nx = cy + dy, cx + dx  # type: ignore
            
            # Check if forward move is valid
            blocked = False
            if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                blocked = True
            elif grid[ny, nx] == 1:
                blocked = True
            elif phase == 2 and terrain is not None:
                slope = max(0, terrain[ny, nx] - terrain[cy, cx])
                if slope > MAX_SLOPE:
                    blocked = True
            
            if blocked:
                # Don't crash! Instead, turn toward goal and try again.
                consecutive_blocked += 1
                if consecutive_blocked > 16:
                    status = "stuck"
                    break
                # Turn toward the goal direction
                dy_g = gy - cy  # type: ignore
                dx_g = gx - cx  # type: ignore
                if abs(dx_g) >= abs(dy_g):
                    desired_hd = EAST if dx_g > 0 else WEST
                else:
                    desired_hd = SOUTH if dy_g > 0 else NORTH
                # Decide left or right turn toward desired heading
                left_dist = (hd - desired_hd) % 4  # type: ignore
                right_dist = (desired_hd - hd) % 4  # type: ignore
                if left_dist <= right_dist:
                    hd = (hd - 1) % 4
                else:
                    hd = (hd + 1) % 4
                path.append((cy, cx, hd))
            else:
                # Move forward successfully
                consecutive_spins = 0
                consecutive_blocked = 0
                cy, cx = ny, nx
                path.append((cy, cx, hd))
        elif action == LEFT:
            hd = (hd - 1) % 4
            consecutive_spins += 1
            path.append((cy, cx, hd))
        elif action == RIGHT:
            hd = (hd + 1) % 4
            consecutive_spins += 1
            path.append((cy, cx, hd))
            
        if consecutive_spins >= 16:
            status = "spinning_failure"
            break
        
    if status == "running":
        status = "max_steps"
        
    return path, status
