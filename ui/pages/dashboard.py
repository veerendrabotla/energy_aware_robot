import streamlit as st # type: ignore
import sys # type: ignore
import os # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.train_ann import load_model # type: ignore

st.set_page_config(page_title="Dashboard", layout="wide")

# Init checks
from ui.session_manager import init_session # type: ignore
init_session()

st.title("🏠 Advanced Dashboard")
st.markdown("Welcome to the isolated Dashboard view.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Maps Generated", st.session_state.maps_generated)
col2.metric("ANN Model Status", "Ready ✅" if st.session_state.model else "Not Trained ❌")
col3.metric("Dataset Size", st.session_state.dataset_rows)
col4.metric("Last Accuracy", f"{st.session_state.last_accuracy}%" if st.session_state.last_accuracy != "--" else "--")

st.markdown("---")
st.info("Navigate using the sidebar to delve deeply into each specific module.")
