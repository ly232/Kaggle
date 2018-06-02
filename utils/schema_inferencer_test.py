import datetime
import numpy as np
import unittest

from schema_inferencer import CsvSchemaInferencer

class TestSchemaInferencer(unittest.TestCase):
  def setUp(self):
    self._csv_schema_inferencer = CsvSchemaInferencer(
      'testdata/schema_inferencer_test.csv')

  def testGetRawHeader(self):
    self.assertEqual(
      ['col' + str(i + 1) for i in range(6)], 
      self._csv_schema_inferencer._raw_header)

  def testGetRawData(self):
    self.assertEqual(
      [[1.0, 'x', '', datetime.datetime(2000, 1, 1, 0, 0), '1234-56-78', 2.3],
       [2.0, 'y', '', datetime.datetime(2001, 2, 2, 0, 0), '1234-56-79', 2.3],
       [2.0, 'z', '', datetime.datetime(2001, 2, 2, 0, 0), '1234-56-79', 2.3]],
      self._csv_schema_inferencer._raw_data)

  def testGetNonCategoricalData(self):
    self.assertEqual(
      [[1.0, datetime.datetime(2000, 1, 1, 0, 0), 2.3],
       [2.0, datetime.datetime(2001, 2, 2, 0, 0), 2.3],
       [2.0, datetime.datetime(2001, 2, 2, 0, 0), 2.3]],
      self._csv_schema_inferencer._non_categorical_data)

  def testGetCategoricalData(self):
    self.assertEqual(
      [[1, 0, 0], [0, 0, 1], [2, 0, 1]],
      self._csv_schema_inferencer._categorical_data)

  def testGetShuffledHeader(self):
    self.assertEqual(
      ['col1', 'col4', 'col6', 'col2', 'col3', 'col5'],
      self._csv_schema_inferencer._shuffled_header)

  def testOneHotEncode(self):
    np.testing.assert_array_equal(
      np.array([[0, 1, 0, 1, 0, 1]]),
      self._csv_schema_inferencer.OneHotEncode([[1, 0, 1]]))

if __name__ == '__main__':
  unittest.main()
