"""Utilities for schema inference related functionalities."""
import csv
import numpy as np

from datetime import datetime
from sklearn import preprocessing

#
# Constants.
#
_DATETIME_PATTERNS = [
  '%Y-%m-%d',
]

#
# Common helper methods.
#
def MaybeGetFloat(value):
  try:
    return float(value)
  except Exception:
    return value

def MaybeGetDate(value):
  for pattern in _DATETIME_PATTERNS:
    try:
      return datetime.strptime(value, pattern)
    except Exception:
      continue
  return value

def IsCategorical(value):
  return type(value) is str

# CSV Schema inferencer to get schema from a CSV file. It notably uses one-hot
# encoding to expand enum-like columns, and attach type annotations to columns
# in the widened version.
class CsvSchemaInferencer(object):

  """Initialized the inferencer with header row and data row.

  Args:
     filename (str): CSV filename.
     target_colname (str): Column name for the target variable.
  """
  def __init__(self, filename, target_colname=None):
    # Header and data extracted from CSV file.
    self._raw_header = None
    self._raw_data = []

    self._raw_train_data = []
    self._raw_test_data = []
    self._raw_target_data = []
    self._raw_no_target_header = []

    with open(filename, 'r') as f:
      csv_reader = csv.reader(f)
      for row in csv_reader:
        if self._raw_header is None:
          self._raw_header = row
        else:
          self._raw_data.append(map(MaybeGetFloat, map(MaybeGetDate, row)))

    if target_colname is not None:
      target_colindex = self._raw_header.index(target_colname)
      self._raw_no_target_header = (
        self._raw_header[:target_colindex] + 
        self._raw_header[target_colindex + 1 :]
      )
      for row in self._raw_data:
        x = [row[i] for i in range(len(row)) if i != target_colindex]
        y = row[target_colindex]
        if row[target_colindex] == '':
          self._raw_test_data.append(x)
        else:
          self._raw_train_data.append(x)
          self._raw_target_data.append([y])

    # Data splitted by non-categorical vs. categorical.
    non_categorical_data, categorical_data, categories_count = (
      self._SplitCategorialColumns())

    # Shuffled header corresponds to the splitted data and one hot encoding.
    self._shuffled_header = []
    if self._raw_train_data:
      non_categorical_indexes = [
        i for i in range(len(self._raw_train_data[0])) 
          if not IsCategorical(self._raw_train_data[0][i])]
      categorical_indexes = [
        i for i in range(len(self._raw_train_data[0])) 
          if IsCategorical(self._raw_train_data[0][i])]
      for i in non_categorical_indexes:
        self._shuffled_header.append(self._raw_no_target_header[i])
      offset = 0
      assert len(categorical_indexes) == len(categories_count)
      for i in categorical_indexes:
        self._shuffled_header.extend(
          [self._raw_no_target_header[i] + '_' + str(j) 
           for j in range(categories_count.values()[offset])]
        )
        offset += 1

    # Apply one-hot-encoding to categorical data.
    self._one_hot_encoder = preprocessing.OneHotEncoder()
    self._one_hot_encoder.fit(categorical_data)

    # Augment non-categorical and categorical data. For categorical data, apply
    # one-hot-encoding.
    encoded_categorical_data = self._one_hot_encoder.transform(
      categorical_data).toarray()
    self._data = {
      'X': np.hstack((non_categorical_data, encoded_categorical_data)),
      'y': np.array(self._raw_target_data),
      'X_schema': self._shuffled_header,
      'y_schema': target_colname,
    }

  def GetData(self):
    return self._data

  """Split by non-categorical and categorical columns.

  ScikitLearn's one hot encoder requires that all categorical columns be in 
  integer format. This method is the helper to convert raw data into that 
  foramt.

  Returns to sets of data that are column-splitted, such that one set consists
  of non-categorical data, the other contains categorical data.
  """
  def _SplitCategorialColumns(self):
    if not self._raw_train_data:
      return []

    # Cluster categorical columns together.
    data = [
      sorted(row, key=IsCategorical)
      for row in self._raw_train_data
    ]
    categorical_column_start_index = None
    for i in range(len(data[0])):
      if IsCategorical(data[0][i]):
        categorical_column_start_index = i
        break

    if categorical_column_start_index is None:
      return data

    catetories_by_column_index = {
      i: set()
      for i in range(categorical_column_start_index, len(data[0]))
    }

    # Non-category data are copied as-is.
    non_category_data = []
    for row in data:
      new_row = []
      for i in range(len(row)):
        if i < categorical_column_start_index:
          new_row.append(row[i])
        else:
          catetories_by_column_index[i].add(row[i])
      non_category_data.append(new_row)

    # Category data are transformed into integer values.
    category_data = []
    for row in data:
      new_row = []
      for i in range(len(row)):
        if i in catetories_by_column_index:
          categories = list(catetories_by_column_index[i])
          new_row.append(categories.index(row[i]))
      category_data.append(new_row)

    # Keep track of number of elements per category.
    categories_count = {
      i: len(catetories_by_column_index[i])
      for i in catetories_by_column_index
    }

    return non_category_data, category_data, categories_count
