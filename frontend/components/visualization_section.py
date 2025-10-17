
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol

def render_visualization_section():
    if not st.session_state.get('prediction_results'):
        st.info("👈 Run a prediction to see visualizations")
        return

    results = st.session_state.prediction_results
    st.markdown("### 📊 Analysis Visualizations")

    tabs = st.tabs(["Attribution", "Structure", "Uncertainty"])

    with tabs[0]:
        if results.get("atom_importance") is not None:
            fig, ax = plt.subplots()
            importance = results["atom_importance"]
            sns.barplot(x=list(range(len(importance))), y=np.abs(importance), ax=ax)
            ax.set_title("Per-Atom Importance (Absolute Value)")
            st.pyplot(fig)
        else:
            st.info("Attribution analysis was not enabled.")

    with tabs[1]:
        smiles = st.session_state.drug_smiles
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(500, 300))
        st.image(img, caption="2D Molecular Structure")

    with tabs[2]:
        if 'confidence_interval' in results:
            score, std = results['score'], results['uncertainty']
            samples = np.random.normal(score, std, 1000)
            fig, ax = plt.subplots()
            sns.histplot(samples, kde=True, ax=ax)
            ax.axvline(score, color='r', linestyle='--', label=f"Mean: {score:.3f}")
            ax.set_title("Prediction Uncertainty Distribution"); ax.legend()
            st.pyplot(fig)
        else:
            st.info("Uncertainty estimation was not enabled.")
