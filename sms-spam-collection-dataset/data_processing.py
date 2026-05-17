"""Utilities to cleans raw Kaggle datasets."""

import pandas as pd

from transformers import GPT2Tokenizer


def clean_sms_spam_collection_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw SMS Spam Collection dataset.

    Args:
        df: The raw DataFrame loaded from the dataset.

    Returns:
        A cleaned DataFrame with unnecessary columns removed and renamed for clarity.
    """
    # Drop unnecessary columns
    unwanted_columns = [col for col in df.columns if col.startswith("Unnamed")]
    for col in unwanted_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Rename columns for clarity
    df = df.rename(columns={"v1": "label", "v2": "message"})

    return df


def tokenize_message(tokenizer: GPT2Tokenizer, df: pd.DataFrame) -> pd.DataFrame:
    """Widens the input dataframe by tokenizing the messages."""
    if "tokens" in df.columns:
        return df
    df["tokens"] = df["message"].apply(lambda m: tokenizer(m)["input_ids"])

    return df
