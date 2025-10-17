
import torch
from transformers import AutoTokenizer, EsmModel
from backend.utils.config import DTIConfig

class ProteinEmbedder:
    _instance = None

    # Use a singleton pattern to avoid loading the model multiple times
    def __new__(cls, config: DTIConfig):
        if cls._instance is None:
            cls._instance = super(ProteinEmbedder, cls).__new__(cls)
            cls._instance.config = config
            cls._instance.device = torch.device(config.device)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(config.protein_embedder_model)
            cls._instance.model = EsmModel.from_pretrained(config.protein_embedder_model).to(cls._instance.device)
            cls._instance.model.eval()
        return cls._instance

    def embed(self, sequence: str):
        # Truncate sequence to be safe with model limits
        sequence = sequence[:1022]
        with torch.no_grad():
            inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True).to(self.device)
            outputs = self.model(**inputs)
            # Use mean pooling of the last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.cpu() # Return on CPU to save GPU memory
