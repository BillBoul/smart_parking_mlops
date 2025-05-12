import pandas as pd

def load_dataset(csv_path):
    """Loads the merged dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
