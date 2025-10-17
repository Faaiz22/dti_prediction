
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def random_split(df, config):
    train_val_df, test_df = train_test_split(df, test_size=config.test_size, random_state=config.random_seed)
    train_df, val_df = train_test_split(train_val_df, test_size=config.val_size/(1-config.test_size), random_state=config.random_seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def scaffold_split(df, config):
    # This is a placeholder for a proper scaffold split implementation using RDKit.
    print("Warning: Using random split as a placeholder for scaffold split.")
    return random_split(df, config)

def plot_scaffold_distribution(train_df, val_df, test_df, save_path):
    plt.figure(figsize=(8, 5))
    plt.bar(['Train', 'Validation', 'Test'], [len(train_df), len(val_df), len(test_df)], color=['blue', 'orange', 'green'])
    plt.title('Dataset Split Sizes')
    plt.ylabel('Number of Samples')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
