import plotly.graph_objects as go
import numpy as np

def plot_3d_terrain(terrain, start, goal, astar_path=None, ann_path=None):
    size = terrain.shape[0]
    
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    xGrid, yGrid = np.meshgrid(x, y)
    
    fig = go.Figure(data=[go.Surface(z=terrain, x=xGrid, y=yGrid, colorscale='Viridis', opacity=0.8)])
    
    sy, sx = start
    gy, gx = goal
    fig.add_trace(go.Scatter3d(x=[sx], y=[sy], z=[terrain[sy, sx]], mode='markers', marker=dict(color='green', size=8), name='Start'))
    fig.add_trace(go.Scatter3d(x=[gx], y=[gy], z=[terrain[gy, gx]], mode='markers', marker=dict(color='red', size=8), name='Goal'))
    
    if astar_path:
        ax = [p[1] for p in astar_path]
        ay = [p[0] for p in astar_path]
        az = [terrain[y, x] for y, x in zip(ay, ax)]
        fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode='lines', line=dict(color='blue', width=4), name='A* Path'))
        
    if ann_path:
        nx = [p[1] for p in ann_path]
        ny = [p[0] for p in ann_path]
        nz = [terrain[y, x] for y, x in zip(ny, nx)]
        fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode='lines', line=dict(color='orange', width=4, dash='dash'), name='ANN Path'))
        
    fig.update_layout(title="3D Terrain Navigation",
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Elevation'))
    return fig
