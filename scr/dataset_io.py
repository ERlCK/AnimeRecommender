import pandas as pd


def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
