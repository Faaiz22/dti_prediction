
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

@dataclass
class DTIConfig:
    # --- Paths and Seeds ---
    data_path: Path = field(default_factory=lambda: Path("data/relationships.tsv"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    device: str = "cuda"
    random_seed: int = 42
    log_level: str = "INFO"

    # --- Data Processing ---
    use_scaffold_split: bool = False
    test_size: float = 0.1
    val_size: float = 0.1
    batch_size: int = 32

    # --- Featurization ---
    protein_embedder_model: str = "facebook/esm2_t6_8M_UR50D"
    num_atom_features: int = 5  # As per molecular featurizer
    num_bond_features: int = 3  # As per molecular featurizer

    # --- Model Architecture ---
    drug_node_dim: int = 128
    drug_edge_dim: int = 64
    drug_hidden_dim: int = 128
    egnn_num_layers: int = 4
    protein_embedding_dim: int = 320  # For esm2_t6_8M
    use_cross_attention: bool = True
    attention_dim: int = 128
    attention_heads: int = 4
    mlp_hidden_dims: list = field(default_factory=lambda: [256])
    dropout_rate: float = 0.2

    # --- Training ---
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 10

    # --- Inference ---
    mc_dropout_samples: int = 25

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    @classmethod
    def load(cls, path: Path):
        with open(path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if 'dir' in k or 'path' in k: data[k] = Path(v)
        return cls(**data)
