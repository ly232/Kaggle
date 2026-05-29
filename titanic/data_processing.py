import pandas as pd
import duckdb
import torch

from jaxtyping import Float
from torch import Tensor


def to_tensors(
    input_df: pd.DataFrame,
) -> tuple[Float[Tensor, "batch_size input_dim"], Float[Tensor, "batch_size 1"] | None]:
    """Transforms the input df into X and y tensors to feed into model.

    Thoughts on the data:
    * Certain columns are likely irrelevant, e.g. PassengerId, Name, Ticket, ...
    * Cabin column is intuitively useful, but most are NaN, so we ignore them too.
    * For the rest, we need to one-hot encode the categorical columns. Note the only non-categorical columns are age and fare.

    Args:
        input_df: The input dataframe, either the train or test df. For train
          df, column "Survived" must exist; else it's a test df.

    Returns:
        A tuple of (X_tensor, y_tensor). If the input df is a test df, then
          y_tensor will be None.
    """
    is_train = "Survived" in input_df.columns
    input_df["Embarked"] = input_df["Embarked"].fillna(input_df["Embarked"].mode()[0])
    # Prune the dataset to only include relevant columns.
    featurized_df = duckdb.sql("""
    SELECT
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    FROM input_df;
    """).df()
    # One-hot encode the categorical columns.
    featurized_df = pd.get_dummies(
        featurized_df,
        columns=[
            "Pclass",
            "Sex",
            "SibSp",
            "Parch",
            "Embarked",
        ],
        drop_first=True,  # Drop the first category to avoid multicollinearity.
        dtype=float,  # Ensure the one-hot encoded columns are of type float.
    )
    print(featurized_df.columns)

    # Data cleansing.
    featurized_df["Age"] = featurized_df["Age"].fillna(featurized_df["Age"].median())
    featurized_df["Fare"] = featurized_df["Fare"].fillna(featurized_df["Fare"].median())

    # Transform df into  input and output tensors.
    X = featurized_df.astype("float32")
    if "Parch_9" not in X.columns:
        assert (
            is_train
        ), "Parch_9 is missing from train data set, so we manually fake the column."
        X["Parch_9"] = 0
    y = input_df["Survived"] if is_train else None
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = (
        torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) if is_train else None
    )
    return X_tensor, y_tensor
