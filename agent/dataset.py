"""Dataset loader for Bitext Customer Service dataset."""

import os
import pandas as pd


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_CSV_PATH = os.path.join(_DATA_DIR, "bitext_customer_service.csv")


def load_dataset() -> pd.DataFrame:
    """Load the Bitext Customer Service dataset.

    Downloads from HuggingFace on first call, then caches locally as CSV.
    Returns a pandas DataFrame with all dataset columns.
    """
    if os.path.exists(_CSV_PATH):
        return pd.read_csv(_CSV_PATH)

    # Download from HuggingFace
    from datasets import load_dataset as hf_load

    ds = hf_load("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = ds.to_pandas()
    os.makedirs(_DATA_DIR, exist_ok=True)
    df.to_csv(_CSV_PATH, index=False)
    return df


def get_dataframe() -> pd.DataFrame:
    """Get the cached DataFrame (singleton pattern)."""
    if not hasattr(get_dataframe, "_df"):
        get_dataframe._df = load_dataset()
    return get_dataframe._df
