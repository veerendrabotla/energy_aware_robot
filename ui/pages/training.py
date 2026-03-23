import streamlit as st # type: ignore
import sys
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.dataset_generator import generate_dataset # type: ignore
from model.train_ann import train_and_save_model # type: ignore
from ui.session_manager import init_session # type: ignore
init_session()

st.set_page_config(page_title="Model Training", layout="wide")

st.title("🧠 Neural Network Training Center")

colA, colB = st.columns([1, 2])
with colA:
    st.markdown("### Generate Massive Dataset")
    st.markdown("Capture perfect heuristic data by running thousands of A* algorithms over random generations.")
    maps_to_gen = st.slider("Number of Maps", 10, 200, 50)
    scenarios = st.slider("Scenarios per Map", 5, 50, 10)
    st.radio("Feature Set Phase", [1, 2], key="global_phase")
    phase_sel = st.session_state.global_phase
    
    if st.button("Build Dataset", width="stretch"):
        with st.spinner("Extracting features and labeling..."):
            X, y, path = generate_dataset(phase=phase_sel, num_maps=maps_to_gen, scenarios_per_map=scenarios)
            st.session_state.temp_X = X
            st.session_state.temp_y = y
            st.session_state.dataset_rows = len(X)
        st.success(f"Built {len(X)} labeled data points!")
        
    st.markdown("---")
    if st.button("Train MLP Neural Network", width="stretch"):
        if "temp_X" in st.session_state:
            with st.spinner("Fitting Adam optimizer to Multilayer Perceptron..."):
                model, score, _ = train_and_save_model(st.session_state.temp_X, st.session_state.temp_y, phase=phase_sel)
                st.session_state.model = model
                st.session_state.last_accuracy = round(score * 100, 2)
            st.success(f"Network convergence successful! Validated Accuracy: {st.session_state.last_accuracy}%")
        else:
            st.error("Missing Dataset! Please build the dataset above first.")

with colB:
    if "temp_y" in st.session_state:
        st.markdown("### Internal Class Distribution")
        fwd = int(np.sum(np.array(st.session_state.temp_y) == 0))
        lft = int(np.sum(np.array(st.session_state.temp_y) == 1))
        rgt = int(np.sum(np.array(st.session_state.temp_y) == 2))
        df_dist = pd.DataFrame({"Actions": ["Forward", "Left", "Right"], "Samples": [fwd, lft, rgt]})
        st.bar_chart(df_dist.set_index("Actions"), color="#ffaa00")
        
        st.markdown("### 🔍 Raw Dataset Transparency File")
        num_features = st.session_state.temp_X.shape[1]
        cols = ["Distance", "Rel_Angle", "Obs_F", "Obs_L", "Obs_R"] if num_features == 5 else ["Distance", "Rel_Angle", "Slope_F", "Slope_L", "Slope_R", "Obs_F", "Obs_L", "Obs_R"]
        
        # Convert to Pandas for reading
        df_dataset = pd.DataFrame(st.session_state.temp_X, columns=cols)
        action_map = {0: "Forward", 1: "Left", 2: "Right"}
        df_dataset["Label_Action"] = [action_map[val] for val in st.session_state.temp_y]
        
        st.dataframe(df_dataset, height=300)
        
        csv_data = df_dataset.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Dataset (CSV)", data=csv_data, file_name=f"energy_robot_dataset_phase_{phase_sel}.csv", mime='text/csv')
    else:
        st.info("Run generation to see transparent statistics here.")
