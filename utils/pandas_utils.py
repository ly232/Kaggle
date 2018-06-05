r"""Helper methods to operate over Pandas data structures. They are mostly
one-liners to help achieve common operations over Kaggle datasets, and are 
mostly inspired from solutions posted online.

See pandas_utils_test.py for example usage.
"""
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer


def DataFrameDifference(df1, df2):
  """Performs set-difference operation on 2 dataframes.

  Inspiared from https://stackoverflow.com/questions/18180763

  Args:
    df1 (pd.DataFrame): DataFrame to subtract from.
    df2 (pd.DataFrame): DataFrame to subtract by.

  Returns: New pd.DataFrame object with rows in df1 but not in df2.
  """
  return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)


def GetNullableColumns(df):
  """Gets all columns that have at least one entry equal to numpy.nan.

  Inspired from https://stackoverflow.com/questions/36226083

  Args:
    df (pd.DataFrame): Pandas DataFrame to inspect.

  Returns: List of strings for the colum names.
  """
  return df.columns[df.isna().any()].tolist()


def ProjectOutNullColumns(df):
  """Projects out all columns that contains at least one null value.

  Args:
    df (pd.DataFrame): Pandas DataFrame to inspect.

  Returns: Pandas DataFrame with a subset of columns in df, such that there
    are no null values in the returned data frame.
  """
  return df[list(set(df.columns) - set(GetNullableColumns(df)))]


def DropColumns(df, cols):
  """Drops colums in a dataframe by name.

  Args:
    df (pd.DataFrame): Pandas DataFrame to inspect.
    cols (list[str]): Names of columns to drop.

  Returns: New data frame with some columns dropped.
  """
  return df.drop(cols, axis=1)


def KeepColumns(df, cols):
  """Keeps the columns cols in df, drop all other columns.

  Args:
    df (pd.DataFrame): Pandas DataFrame to inspect.
    cols (list[str]): Names of columns to keep.

  Returns: New data frame with some columns dropped.
  """
  cols_to_drop = list(set(df.columns) - set(cols))
  return DropColumns(df, cols_to_drop)


def AugmentColumns(df, cols_with_defaults):
  """Augments new columns with default values.

  Args:
    df (pd.DataFrame): Pandas DataFrame to inspect.
    cols_with_defaults (dict[str, object]): Maps from column name to default 
      value.

  Returns: New data frame with augmented columns with default values.
  """
  new_df = df.copy()
  for col in cols_with_defaults:
    new_df[col] = cols_with_defaults[col]
  return new_df


def OneHotEncode(df):
  """Applies one-hot-encoding for non-numerical columns.

  Inspired from:
  -  https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
  -  https://stackoverflow.com/questions/21720022

  Args:
    df (pd.DataFrame): DataFrame to apply one-hot-encoding.

  Returns: One-hot-encoded pd.DataFrame.
  """
  vec = DictVectorizer()
  cols = df.select_dtypes(include=[np.object]).columns

  mkdict = lambda row: dict((col, row[col]) for col in cols)
  vec_data = pd.DataFrame(
    vec.fit_transform(df[cols].apply(mkdict, axis=1)).toarray())
  vec_data.columns = vec.get_feature_names()
  vec_data.index = df.index
  
  df = df.drop(cols, axis=1)
  df = df.join(vec_data)
  return df
