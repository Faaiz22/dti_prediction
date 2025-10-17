
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch_geometric.data import Batch
from backend.utils.config import DTIConfig
from backend.utils.logger import setup_logger

class DTIExplainer:
    def __init__(self, model, config: DTIConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.logger = setup_logger("explainer", level=config.log_level)
        self.model.to(self.device).eval()

    def explain_prediction(self, graph, protein_emb, smiles: str = "", sequence: str = ""):
        graph_batch = Batch.from_data_list([graph]).to(self.device)
        protein_emb_batch = protein_emb.unsqueeze(0).to(self.device)

        def model_forward(node_features, prot_embedding):
            # Captum requires a forward function that accepts tensors.
            # We reconstruct the graph batch inside.
            temp_graph = graph_batch.clone()
            temp_graph.x = node_features
            logits, _ = self.model(temp_graph, prot_embedding)
            return logits

        ig = IntegratedGradients(model_forward)
        baseline_graph_x = torch.zeros_like(graph_batch.x)
        baseline_protein = torch.zeros_like(protein_emb_batch)

        try:
            attributions = ig.attribute(
                (graph_batch.x, protein_emb_batch),
                baselines=(baseline_graph_x, baseline_protein),
                return_convergence_delta=False,
                n_steps=50
            )
            atom_importance = attributions[0].sum(dim=1).cpu().numpy()
        except Exception as e:
            self.logger.warning(f"IG attribution failed: {e}. Attributions will be None.")
            atom_importance = None

        with torch.no_grad():
            logits, attention_weights = self.model(graph_batch, protein_emb_batch)
            prediction = torch.sigmoid(logits).item()

        return {
            "prediction": prediction,
            "atom_importance": atom_importance,
            "attention_weights": attention_weights.cpu().numpy() if attention_weights is not None else None,
            "smiles": smiles, "sequence": sequence
        }

    def generate_report(self, explanation: Dict, save_path: Optional[Path] = None):
        report_lines = ["="*60, "DTI PREDICTION EXPLANATION REPORT", "="*60,
                        f"Prediction Score: {explanation['prediction']:.4f}"]
        if explanation.get("atom_importance") is not None:
            report_lines.append("\n--- Top 5 Contributing Atoms (by Importance) ---")
            top_indices = np.argsort(np.abs(explanation["atom_importance"]))[-5:][::-1]
            for i in top_indices:
                report_lines.append(f"  Atom {i}: Importance = {explanation['atom_importance'][i]:.4f}")

        report = "\n".join(report_lines)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(report)
        return report
