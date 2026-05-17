"""Utilities to cleans raw Kaggle datasets."""

import pandas as pd


def clean_sms_spam_collection_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw SMS Spam Collection dataset.

    Args:
        df: The raw DataFrame loaded from the dataset.

    Returns:
        A cleaned DataFrame with unnecessary columns removed and renamed for clarity.
    """
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

    # Rename columns for clarity
    df = df.rename(columns={"v1": "label", "v2": "message"})

    return df
