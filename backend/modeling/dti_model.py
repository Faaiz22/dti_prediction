
from typing import Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from backend.utils.config import DTIConfig

class EGNNLayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='mean')
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, pos):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)

    def message(self, x_i, x_j, edge_attr, pos_i, pos_j):
        dist = torch.norm(pos_i - pos_j, dim=1, keepdim=True)
        msg_input = torch.cat([x_i, x_j, edge_attr, dist], dim=1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.norm(x + self.update_mlp(update_input))

class CrossAttention(nn.Module):
    def __init__(self, drug_dim: int, protein_dim: int, attn_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(protein_dim, attn_dim)
        self.to_kv = nn.Linear(drug_dim, attn_dim * 2)
        self.out_proj = nn.Linear(attn_dim, attn_dim)

    def forward(self, drug_emb, protein_emb):
        q = self.to_q(protein_emb).view(protein_emb.size(0), self.num_heads, 1, self.head_dim)
        k, v = self.to_kv(drug_emb).chunk(2, dim=-1)
        k = k.view(drug_emb.size(0), self.num_heads, -1, self.head_dim)
        v = v.view(drug_emb.size(0), self.num_heads, -1, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v).view(protein_emb.size(0), -1)
        return self.out_proj(attn_output), attn_weights

class DTIModel(nn.Module):
    def __init__(self, config: DTIConfig):
        super().__init__()
        self.config = config
        self.node_embedding = nn.Linear(config.num_atom_features, config.drug_node_dim)
        self.edge_embedding = nn.Linear(config.num_bond_features, config.drug_edge_dim)

        self.egnn_layers = nn.ModuleList([
            EGNNLayer(config.drug_node_dim if i == 0 else config.drug_hidden_dim, config.drug_edge_dim, config.drug_hidden_dim)
            for i in range(config.egnn_num_layers)
        ])

        if config.use_cross_attention:
            self.cross_attention = CrossAttention(config.drug_hidden_dim, config.protein_embedding_dim, config.attention_dim, config.attention_heads)
            mlp_input_dim = config.attention_dim
        else:
            self.cross_attention = None
            mlp_input_dim = config.drug_hidden_dim + config.protein_embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, config.mlp_hidden_dims[0]), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_hidden_dims[0], 1)
        )

    def forward(self, graph, protein_emb):
        x, edge_index, edge_attr, pos, batch = graph.x, graph.edge_index, graph.edge_attr, graph.pos, graph.batch
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr, pos)

        drug_emb = global_mean_pool(x, batch)

        if self.cross_attention:
            combined, attn_weights = self.cross_attention(drug_emb, protein_emb)
        else:
            combined = torch.cat([drug_emb, protein_emb], dim=1)
            attn_weights = None

        logits = self.mlp(combined).squeeze(-1)
        return logits, attn_weights

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        config = DTIConfig(**checkpoint["config"])
        model = cls(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint
