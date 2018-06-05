import pandas_utils

import pandas as pd
import numpy as np
import unittest

from pandas.util.testing import assert_frame_equal


class TestSchemaInferencer(unittest.TestCase):

  def testDataFrameDifference(self):
    df1 = pd.DataFrame({
      'foo': [1, 2, 3],
      'bar': [4, 5, 6]
    })
    df2 = pd.DataFrame({
      'foo': [2, 7],
      'bar': [5, 8]
    })
    diff = pandas_utils.DataFrameDifference(df1, df2)
    expected = pd.DataFrame({
      'foo': [1, 3],
      'bar': [4, 6]
    })
    expected.index = [0, 2]
    assert_frame_equal(expected, diff)


  def testGetNullableColumns(self):
    df = pd.DataFrame({
      'foo': [1, 2, np.nan],
      'bar': [3, 4, 5]
    })
    self.assertEqual(['foo'], pandas_utils.GetNullableColumns(df))


  def testProjectOutNullColumns(self):
    df = pd.DataFrame({
      'foo': [1, 2, np.nan],
      'bar': [3, 4, 5]
    })
    assert_frame_equal(pd.DataFrame({
      'bar': [3, 4, 5]
    }), pandas_utils.ProjectOutNullColumns(df))


  def testDropColumns(self):
    df = pd.DataFrame({
      'foo': [1, 2, np.nan],
      'bar': [3, 4, 5]
    })
    assert_frame_equal(pd.DataFrame({
      'bar': [3, 4, 5]
    }), pandas_utils.DropColumns(df, ['foo']))


  def testKeepColumns(self):
    df = pd.DataFrame({
      'foo': [1, 2, np.nan],
      'bar': [3, 4, 5]
    })
    assert_frame_equal(pd.DataFrame({
      'foo': [1, 2, np.nan]
    }), pandas_utils.KeepColumns(df, ['foo']))


  def testAugmentColumns(self):
    df = pd.DataFrame({
      'foo': [1, 2, np.nan],
      'bar': [3, 4, 5]
    })
    assert_frame_equal(pd.DataFrame({
      'foo': [1, 2, np.nan],
      'bar': [3, 4, 5],
      'baz': [0, 0, 0]
    }).sort_index(axis=1),
    pandas_utils.AugmentColumns(df, {'baz': 0}).sort_index(axis=1))


  def testOneHotEncode(self):
    df = pd.DataFrame({
      'foo': [1, 2, 3],
      'bar': ['a', 'b', 'a']
    })
    df_encoded = pandas_utils.OneHotEncode(df)
    assert_frame_equal(pd.DataFrame({
      'bar=a': [1.0, 0.0, 1.0],
      'bar=b': [0.0, 1.0, 0.0],
      'foo': [1, 2, 3],
    }).sort_index(axis=1), df_encoded.sort_index(axis=1))


if __name__ == '__main__':
  unittest.main()