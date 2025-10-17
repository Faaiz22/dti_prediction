
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from backend.utils.config import DTIConfig

class DTIPredictor:
    def __init__(self, model, config: DTIConfig, molecular_featurizer, protein_embedder):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.mol_featurizer = molecular_featurizer
        self.prot_embedder = protein_embedder
        self.model.to(self.device).eval()

    def _predict_with_uncertainty(self, graph_batch, protein_emb, n_samples):
        self.model.train()  # Enable dropout layers
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits, _ = self.model(graph_batch, protein_emb)
                predictions.append(torch.sigmoid(logits).item())
        self.model.eval()  # Disable dropout layers
        return np.array(predictions)

    def predict(self, smiles: str, sequence: str, return_uncertainty: bool = True, return_attention: bool = False):
        graph = self.mol_featurizer.smiles_to_graph(smiles)
        if graph is None: return {"error": "Invalid SMILES"}
        protein_emb = self.prot_embedder.embed(sequence)

        graph_batch = Batch.from_data_list([graph]).to(self.device)
        protein_emb_batch = protein_emb.unsqueeze(0).to(self.device)

        if return_uncertainty and self.config.mc_dropout_samples > 0:
            predictions = self._predict_with_uncertainty(graph_batch, protein_emb_batch, self.config.mc_dropout_samples)
            mean_score = predictions.mean()
            std_score = predictions.std()
            ci = (mean_score - 1.96 * std_score, mean_score + 1.96 * std_score)
            return {"score": float(mean_score), "uncertainty": float(std_score), "confidence_interval": ci}
        else:
            with torch.no_grad():
                logits, attention = self.model(graph_batch, protein_emb_batch)
                score = torch.sigmoid(logits).item()
            result = {"score": score}
            if return_attention and attention is not None:
                result["attention_weights"] = attention.cpu().numpy()
            return result
