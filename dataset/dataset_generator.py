import numpy as np
import math
import os
from config import *
from environment.map_generator import generate_2d_map, get_valid_start_goal
from environment.terrain_generator import generate_3d_terrain
from planning.astar import astar_path
import pickle

def extract_features(grid, terrain, state, goal, phase=2):
    cy, cx, hd = state
    gy, gx = goal
    
    dist = math.sqrt((cy - gy)**2 + (cx - gx)**2)
    
    angle_to_goal_abs = math.degrees(math.atan2(gy - cy, gx - cx))
    heading_angles = {EAST: 0, SOUTH: 90, WEST: 180, NORTH: -90}
    rel_angle = (angle_to_goal_abs - heading_angles[hd]) % 360
    if rel_angle > 180:
        rel_angle -= 360
        
    def is_obs(dy, dx):
        ny, nx = cy + dy, cx + dx
        if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
            return 1.0
        return float(grid[ny, nx])
        
    def get_slope(dy, dx):
        if terrain is None: return 0.0
        ny, nx = cy + dy, cx + dx
        if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
            return 0.0
        return max(0.0, terrain[ny, nx] - terrain[cy, cx])

    dy_f, dx_f = MOVES[hd]
    hd_l = (hd - 1) % 4
    dy_l, dx_l = MOVES[hd_l]
    hd_r = (hd + 1) % 4
    dy_r, dx_r = MOVES[hd_r]
    
    obs_f = is_obs(dy_f, dx_f)
    obs_l = is_obs(dy_l, dx_l)
    obs_r = is_obs(dy_r, dx_r)
    
    if phase == 1:
        return [dist, rel_angle, obs_f, obs_l, obs_r]
        
    slope_f = get_slope(dy_f, dx_f)
    slope_l = get_slope(dy_l, dx_l)
    slope_r = get_slope(dy_r, dx_r)
    
    if slope_f > MAX_SLOPE: obs_f = 1.0
    if slope_l > MAX_SLOPE: obs_l = 1.0
    if slope_r > MAX_SLOPE: obs_r = 1.0
    
    return [dist, rel_angle, slope_f, slope_l, slope_r, obs_f, obs_l, obs_r]

def get_action_from_states(s1, s2):
    _, _, hd1 = s1
    _, _, hd2 = s2
    if hd1 == hd2:
        return FORWARD
    elif hd2 == (hd1 - 1) % 4:
        return LEFT
    elif hd2 == (hd1 + 1) % 4:
        return RIGHT
    return FORWARD

def generate_dataset(phase=2, num_maps=NUM_MAPS_TRAIN, scenarios_per_map=SCENARIOS_PER_MAP, size=DEFAULT_MAP_SIZE):
    X = []
    y = []
    
    for _ in range(num_maps):
        grid = generate_2d_map(size, density=DEFAULT_OBSTACLE_DENSITY)
        terrain = generate_3d_terrain(size, terrain_type="mixed") if phase == 2 else np.zeros((size, size))
        
        for _ in range(scenarios_per_map):
            start, goal = get_valid_start_goal(grid)
            path = astar_path(grid, terrain, start, goal, phase=phase)
            if path and len(path) > 1:
                for i in range(len(path) - 1):
                    s_curr = path[i]
                    s_next = path[i+1]
                    action = get_action_from_states(s_curr, s_next)
                    features = extract_features(grid, terrain, s_curr, goal, phase=phase)
                    X.append(features)
                    y.append(action)
                    
    X = np.array(X)
    y = np.array(y)
    
    dataset_path = os.path.join(DATASET_DIR, f"dataset_phase_{phase}.pkl")
    with open(dataset_path, 'wb') as f:
        pickle.dump((X, y), f)
        
    return X, y, dataset_path
