
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from backend.utils.config import DTIConfig

class MolecularFeaturizer:
    def __init__(self, config: DTIConfig):
        self.config = config

    def _get_atom_features(self, atom):
        return [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetTotalNumHs(),
            float(atom.GetIsAromatic()), atom.GetFormalCharge()
        ]

    def _get_bond_features(self, bond):
        return [
            bond.GetBondTypeAsDouble(), float(bond.GetIsAromatic()), float(bond.IsInRing())
        ]

    def smiles_to_graph(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=42) == -1: # Try 3D
                AllChem.Compute2DCoords(mol) # Fallback to 2D
            AllChem.UFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            if pos.shape[1] == 2: # Pad 2D to 3D
                pos = F.pad(pos, (0, 1), "constant", 0)

            atom_feats = torch.tensor([self._get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
                bond_f = self._get_bond_features(bond)
                edge_attrs.extend([bond_f, bond_f])

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

            return Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_attr, pos=pos, smiles=smiles)
        except Exception:
            return None
