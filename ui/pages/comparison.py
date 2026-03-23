import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore
import pandas as pd # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.metrics import compute_metrics # type: ignore

st.set_page_config(page_title="Comparison", layout="wide")
st.title("⚖️ Heuristic vs AI Head-to-Head")

from ui.session_manager import init_session # type: ignore
init_session()

if st.session_state.get("astar_path") or st.session_state.get("ann_path"):
    metrics_a = compute_metrics(st.session_state.astar_path, "success", st.session_state.terrain, phase=st.session_state.global_phase) if st.session_state.get("astar_path") else None
    
    ann_path = st.session_state.get("ann_path")
    status_ann = "success" if (ann_path and ann_path[-1][:2] == st.session_state.goal) else "failed"
    metrics_n = compute_metrics(ann_path, status_ann, st.session_state.terrain, phase=st.session_state.global_phase) if ann_path else None
    
    data = []
    if metrics_a: data.append({"Alg": "A*", "Energy": metrics_a['total_energy'], "Length": metrics_a['path_length'], "Turns": metrics_a['turns'], "Status": "Optimal"})
    if metrics_n: data.append({"Alg": "ANN", "Energy": metrics_n['total_energy'], "Length": metrics_n['path_length'], "Turns": metrics_n['turns'], "Status": metrics_n['status']})
    
    if data:
        df = pd.DataFrame(data)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Cost Metrics Table")
            st.dataframe(df.style.highlight_min(subset=['Energy', 'Length', 'Turns'], color='lightgreen', axis=0))
        with c2:
            st.markdown("### Visual Cost Overview")
            st.bar_chart(df.set_index('Alg')[['Energy', 'Length', 'Turns']])
else:
    st.info("Run simulations to see the comparative overhead!")
