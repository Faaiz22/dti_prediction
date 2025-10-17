
import streamlit as st
from frontend.interface import dti_backend_interface as dti_api

def render_input_section():
    st.markdown("### 1. Input Molecular Data")
    st.session_state.drug_smiles = st.text_input("Drug SMILES String", st.session_state.drug_smiles, placeholder="e.g., CCO")
    st.session_state.protein_sequence = st.text_area("Protein Target Sequence", st.session_state.protein_sequence, height=200, placeholder="e.g., MVLSPADK...")

    if st.session_state.drug_smiles:
        if dti_api.validate_inputs(st.session_state.drug_smiles, "A"*20)[0]:
            with st.expander("🔬 Molecular Properties"):
                st.dataframe(dti_api.get_molecular_properties(st.session_state.drug_smiles), use_container_width=True)
