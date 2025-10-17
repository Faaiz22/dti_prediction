
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm.auto import tqdm

class DTIDataset(Dataset):
    def __init__(self, df, mol_featurizer, prot_embedder, precompute=True):
        self.df = df
        self.mol_featurizer = mol_featurizer
        self.prot_embedder = prot_embedder
        self.precomputed_data = []
        if precompute:
            self._precompute_features()

    def _precompute_features(self):
        print("Pre-computing features for the dataset...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Featurizing"):
            graph = self.mol_featurizer.smiles_to_graph(row['entity1_id'])
            protein_emb = self.prot_embedder.embed(row['entity2_id'])
            label = torch.tensor([row['label']], dtype=torch.float)
            if graph is not None and protein_emb is not None:
                self.precomputed_data.append({'graph': graph, 'protein': protein_emb, 'labels': label})

    def __len__(self):
        return len(self.precomputed_data)

    def __getitem__(self, idx):
        return self.precomputed_data[idx]

def custom_collate(batch):
    graphs = [item['graph'] for item in batch if item is not None]
    if not graphs: return None

    proteins = torch.stack([item['protein'] for item in batch if item is not None])
    labels = torch.cat([item['labels'] for item in batch if item is not None])
    return {"graph": Batch.from_data_list(graphs), "protein": proteins, "labels": labels}

def create_dataloaders(train_df, val_df, test_df, config, mol_featurizer, prot_embedder):
    train_ds = DTIDataset(train_df, mol_featurizer, prot_embedder)
    val_ds = DTIDataset(val_df, mol_featurizer, prot_embedder)
    test_ds = DTIDataset(test_df, mol_featurizer, prot_embedder)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=custom_collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, collate_fn=custom_collate, num_workers=2)
    return train_loader, val_loader, test_loader
