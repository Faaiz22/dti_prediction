
from rdkit import Chem
from rdkit.Chem import Descriptors

def validate_inputs(smiles: str, sequence: str):
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        return False, "Invalid SMILES string"
    if not sequence or len(sequence) < 10:
        return False, "Protein sequence is too short"
    return True, "Valid"

def get_molecular_properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f}",
        "LogP": f"{Descriptors.MolLogP(mol):.2f}",
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
    }
