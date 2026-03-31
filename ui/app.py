import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore
import time # type: ignore
import pandas as pd # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.map_generator import generate_2d_map, get_valid_start_goal # type: ignore
from environment.terrain_generator import generate_3d_terrain # type: ignore
from planning.astar import astar_path # type: ignore
from dataset.dataset_generator import generate_dataset # type: ignore
from model.train_ann import train_and_save_model, load_model # type: ignore
from navigation.run_navigation import run_ann_navigation # type: ignore
from evaluation.metrics import compute_metrics # type: ignore
from plots.plot_2d import plot_2d_grid # type: ignore
from plots.plot_3d import plot_3d_terrain # type: ignore

st.set_page_config(page_title="Energy-Aware Robot", layout="wide")

from ui.session_manager import init_session # type: ignore
init_session()

st.title("🤖 Energy-Aware Robot Navigation System")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controls")
    size = st.slider("Map Size", 10, 50, 20)
    density = st.slider("Obstacle Density", 0.0, 0.5, 0.2)
    terrain_type = st.selectbox("Terrain Type", ["flat", "hills", "steep", "mixed (flat, hills, steep)"])
    st.caption("Hills possess gradual slopes, whereas steep terrain contains sharp elevation spikes.")
    
    if terrain_type == "flat":
        st.session_state.global_phase = 1
        st.radio("Simulation Phase", [1, 2], key="global_phase_fake", disabled=True, index=0, format_func=lambda x: "Phase 1 (2D Only)" if x == 1 else "Phase 2 (2D & 3D Slopes)")
    else:
        st.radio("Simulation Phase", [1, 2], key="global_phase", format_func=lambda x: "Phase 1 (2D Only)" if x == 1 else "Phase 2 (2D & 3D Slopes)")
        
    phase = st.session_state.global_phase

    if st.button("🗺️ Generate Map (or) Terrain", width="stretch"):
        st.session_state.grid = generate_2d_map(size, density)
        st.session_state.terrain = generate_3d_terrain(size, terrain_type)
        st.session_state.start, st.session_state.goal = get_valid_start_goal(st.session_state.grid)
        st.session_state.astar_path = None
        st.session_state.ann_path = None
        
    if st.button("🏃 Run A* Algorithm", width="stretch"):
        if st.session_state.grid is not None:
            path = astar_path(st.session_state.grid, st.session_state.terrain, st.session_state.start, st.session_state.goal, phase=phase)
            if path is None:
                st.error("No valid A* path found! Please regenerate map.")
            else:
                st.session_state.astar_path = path
                st.success("A* Path found!")
        else:
            st.error("Generate a map first.")
                
    if st.button("🧠 Train ANN Model", width="stretch"):
        with st.spinner("Generating ~20,000 samples for dataset..."):
            X, y, path = generate_dataset(phase=phase, num_maps=50, scenarios_per_map=10) # Reduced slightly for live demo speed
        st.success(f"Dataset generated! (Rows: {len(X)})")
        
        with st.spinner("Training Model..."):
            model, score, _ = train_and_save_model(X, y, phase)
            st.session_state.model = model
        st.success(f"Model trained! Accuracy: {score*100:.2f}%")
        
    if st.button("🤖 Run ANN Navigation", width="stretch"):
        if st.session_state.grid is None:
            st.error("Generate a map first.")
        elif st.session_state.model is None:
            st.error("Train or load a model first.")
        else:
            expected_f = 5 if phase == 1 else 8
            if hasattr(st.session_state.model, 'n_features_in_') and st.session_state.model.n_features_in_ != expected_f:
                st.error(f"Model mismatch! Model expects {st.session_state.model.n_features_in_} features. Please retrain under Phase {phase}.")
            else:
                try:
                    path, status = run_ann_navigation(st.session_state.grid, st.session_state.terrain, st.session_state.start, st.session_state.goal, st.session_state.model, phase=phase)
                    st.session_state.ann_path = path
                    if status == "success": st.success("ANN reached goal!")
                    else: st.warning(f"ANN Navigation failed: {status}")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

with col2:
    tab1, tab2, tab3 = st.tabs(["🗺️ Visualizations", "📊 Metrics Comparison", "📝 Info"])
    
    with tab1:
        if st.session_state.grid is not None:
            cA, cB = st.columns(2)
            with cA:
                fig2d = plot_2d_grid(st.session_state.grid, st.session_state.start, st.session_state.goal, st.session_state.astar_path, st.session_state.ann_path, terrain=st.session_state.terrain)
                st.pyplot(fig2d)
            with cB:
                show_3d = st.checkbox("Show 3D Preview", value=(phase == 2), key="app_3d_toggle")
                if show_3d and st.session_state.terrain is not None:
                    fig3d = plot_3d_terrain(st.session_state.terrain, st.session_state.start, st.session_state.goal, st.session_state.astar_path, st.session_state.ann_path)
                    st.plotly_chart(fig3d, width="stretch")
                elif not show_3d:
                    st.info("Toggle the checkbox above to preview 3D terrain.")
    
    with tab2:
        if st.session_state.astar_path or st.session_state.ann_path:
            metrics_a = compute_metrics(st.session_state.astar_path, "success", st.session_state.terrain, phase) if st.session_state.astar_path else None
            metrics_n = compute_metrics(st.session_state.ann_path, "success" if (st.session_state.ann_path and st.session_state.ann_path[-1][:2] == st.session_state.goal) else "failed", st.session_state.terrain, phase) if st.session_state.ann_path else None
            
            data = []
            if metrics_a: data.append({"Algorithm": "A*", "Total Energy": metrics_a['total_energy'], "Path Length": metrics_a['path_length'], "Turns": metrics_a['turns'], "Status": metrics_a['status']})
            if metrics_n: data.append({"Algorithm": "ANN", "Total Energy": metrics_n['total_energy'], "Path Length": metrics_n['path_length'], "Turns": metrics_n['turns'], "Status": metrics_n['status']})
            
            if data:
                df = pd.DataFrame(data)
                st.table(df)
                
                # Simple bar chart
                st.bar_chart(df.set_index('Algorithm')[['Total Energy', 'Path Length', 'Turns']])

    with tab3:
        st.markdown('''### Energy Model
- Forward distance = 1 unit per cell
- Rotating 90° = 5 units
- Moving uphill = 10 units per height diff
- Slopes higher than MAX_SLOPE (0.5) behave like pure walls!''')
