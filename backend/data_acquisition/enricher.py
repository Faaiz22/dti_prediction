
import pandas as pd

class DataEnricher:
    def __init__(self, config):
        self.config = config

    def create_dataset(self):
        # This function now filters the dataset for gene-drug prediction
        # as requested in the problem description.
        try:
            df = pd.read_csv(self.config.data_path, sep='\t')

            # Filter for gene-chemical pairs
            df_genes = df[df['entity1_type'] == 'gene']
            df_gene_drug = df_genes[df_genes['entity2_type'] == 'chemical']

            # Create a binary label (assuming 'association' means binding)
            df_gene_drug['label'] = 1 

            print(f"Loaded and filtered dataset. Kept {len(df_gene_drug)} gene-chemical pairs.")
            return df_gene_drug[['entity1_id', 'entity2_id', 'label']].copy()
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.config.data_path}. Using dummy data.")
            return pd.DataFrame({
                'entity1_id': ['CCO', 'c1ccccc1'], # SMILES
                'entity2_id': ['MVLSPADK', 'MVLSPADKTNVKAAW'], # Sequences
                'label': [1, 0]
            })
