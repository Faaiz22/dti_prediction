
import streamlit as st
from frontend.utils.session_state import load_demo_data

def render_sidebar():
    st.image("https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphadrug-header.gif", use_column_width=True)
    st.markdown("## ⚙️ Settings")

    if st.button("Load Demo (Erlotinib + EGFR)", use_container_width=True):
        load_demo_data()

    st.session_state.use_attribution = st.toggle("Enable Attribution Analysis (IG)", value=True)
    st.session_state.use_mc_dropout = st.toggle("Enable Uncertainty Estimation", value=True)
    if st.session_state.use_mc_dropout:
        st.session_state.mc_samples = st.slider("MC Samples", 10, 100, 25, 5)
