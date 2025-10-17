
import streamlit as st
from frontend.components import (
    render_sidebar, render_input_section,
    render_prediction_section, render_visualization_section
)
from frontend.utils.session_state import initialize_session_state

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="DTI-EGNN Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state variables
    initialize_session_state()

    # Render UI components
    with st.sidebar:
        render_sidebar()

    st.title("🔬 Drug-Target Interaction (DTI) Predictor")
    st.markdown("An interface to predict binding affinity using an E(n)-Equivariant GNN and Cross-Attention model.")

    # Main layout
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        render_input_section()
        render_prediction_section()

    with col2:
        render_visualization_section()

if __name__ == "__main__":
    main()
