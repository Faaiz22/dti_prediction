
import streamlit as st
import numpy as np
from pathlib import Path
from frontend.interface import dti_backend_interface as dti_api
from backend.modeling.dti_model import DTIModel
from backend.utils.config import DTIConfig
from backend.featurization.molecular import MolecularFeaturizer
from backend.featurization.protein import ProteinEmbedder
from backend.inference.dti_inference import DTIPredictor
from backend.inference.dti_explainability import DTIExplainer

@st.cache_resource
def load_model_and_config():
    model_path = Path("checkpoints/best_model_final.pt")
    if not model_path.exists():
        st.error(f"Model checkpoint not found at {model_path}. Please train a model first.")
        return None, None
    model, checkpoint = DTIModel.load_checkpoint(model_path)
    config = DTIConfig(**checkpoint["config"])
    return model, config

def run_prediction():
    smiles = st.session_state.drug_smiles
    sequence = ''.join(st.session_state.protein_sequence.split())
    if not dti_api.validate_inputs(smiles, sequence)[0]:
        st.error("Invalid inputs.")
        return

    model, config = st.session_state.model, st.session_state.config
    if not model: return

    with st.spinner("Running prediction..."):
        config.mc_dropout_samples = st.session_state.mc_samples if st.session_state.use_mc_dropout else 0

        mol_featurizer = MolecularFeaturizer(config)
        prot_embedder = ProteinEmbedder(config) # Uses singleton

        predictor = DTIPredictor(model, config, mol_featurizer, prot_embedder)
        results = predictor.predict(smiles, sequence)

        if st.session_state.use_attribution:
            explainer = DTIExplainer(model, config)
            graph = mol_featurizer.smiles_to_graph(smiles)
            protein_emb = prot_embedder.embed(sequence)
            explanation = explainer.explain_prediction(graph, protein_emb)
            results.update(explanation)

        st.session_state.prediction_results = results

def render_prediction_section():
    st.markdown("### 2. Run Prediction")
    if 'model' not in st.session_state or not st.session_state.model:
        st.session_state.model, st.session_state.config = load_model_and_config()

    if st.button("🚀 Predict Binding Affinity", type="primary", use_container_width=True, disabled=not st.session_state.model):
        run_prediction()

    if st.session_state.get("prediction_results"):
        results = st.session_state.prediction_results
        st.markdown("### 3. Prediction Results")
        score = results['score']
        st.metric("Binding Probability", f"{score:.4f}", f"{'Binding' if score > 0.5 else 'Non-Binding'}")
        if 'uncertainty' in results:
            st.metric("Uncertainty (Std Dev)", f"{results['uncertainty']:.4f}")
