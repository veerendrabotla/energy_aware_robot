import plotly.graph_objects as go # type: ignore
import numpy as np # type: ignore

def plot_3d_terrain(terrain, start, goal, astar_path=None, ann_path=None):
    size = terrain.shape[0]
    
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    xGrid, yGrid = np.meshgrid(x, y)
    
    # Auto-detect terrain type for the title
    max_elev = float(np.max(terrain))
    if max_elev < 0.01:
        terrain_label = "Flat Terrain (Zero Elevation)"
    elif max_elev < 1.5:
        terrain_label = "Hills Terrain (Gradual Slopes)"
    elif max_elev < 3.0:
        terrain_label = "Steep Terrain (Sharp Elevation Changes)"
    else:
        terrain_label = "Mixed Terrain (Flat + Hills + Steep)"
    
    fig = go.Figure(data=[go.Surface(
        z=terrain, x=xGrid, y=yGrid,
        colorscale='Earth',
        opacity=0.85,
        colorbar={'title': 'Elevation', 'ticksuffix': ' m'} # type: ignore
    )])
    
    sy, sx = start
    gy, gx = goal
    fig.add_trace(go.Scatter3d(x=[sx], y=[sy], z=[terrain[sy, sx] + 0.1], mode='markers', marker={'color': 'lime', 'size': 10, 'symbol': 'diamond'}, name='Start')) # type: ignore
    fig.add_trace(go.Scatter3d(x=[gx], y=[gy], z=[terrain[gy, gx] + 0.1], mode='markers', marker={'color': 'red', 'size': 10, 'symbol': 'diamond'}, name='Goal')) # type: ignore
    
    if astar_path:
        ax = [p[1] for p in astar_path]
        ay = [p[0] for p in astar_path]
        az = [terrain[y, x] + 0.05 for y, x in zip(ay, ax)]
        fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode='lines', line={'color': 'blue', 'width': 5}, name='A* Path')) # type: ignore
        
    if ann_path:
        nx = [p[1] for p in ann_path]
        ny = [p[0] for p in ann_path]
        nz = [terrain[y, x] + 0.05 for y, x in zip(ny, nx)]
        fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode='lines', line={'color': 'orange', 'width': 5, 'dash': 'dash'}, name='ANN Path')) # type: ignore
        
    fig.update_layout(
        title=f"3D Terrain Navigation — {terrain_label}",
        scene={
            'xaxis_title': 'X Axis',
            'yaxis_title': 'Y Axis',
            'zaxis_title': 'Elevation (m)',
            'zaxis': {'range': [0, 4]},  # Fixed Z range so flat truly looks flat
        },
        template='plotly_white',
        showlegend=True
    )
    return fig
