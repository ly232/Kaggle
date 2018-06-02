import datetime
import numpy as np
import unittest

from schema_inferencer import CsvSchemaInferencer

class TestSchemaInferencer(unittest.TestCase):
  def setUp(self):
    self._csv_schema_inferencer = CsvSchemaInferencer(
      'testdata/schema_inferencer_test.csv', 'col3')

  def testGetRawData(self):
    self.assertEqual(
      ['col' + str(i + 1) for i in range(6)], 
      self._csv_schema_inferencer._raw_header)

    self.assertEqual(
      [
       [1.0, 'x', '', datetime.datetime(2000, 1, 1, 0, 0), '1234-56-78', 2.3],
       [2.0, 'y', 1.0, datetime.datetime(2001, 2, 2, 0, 0), '1234-56-79', 2.3],
       [2.0, 'z', 0.0, datetime.datetime(2001, 2, 2, 0, 0), '1234-56-79', 2.3]
      ],
      self._csv_schema_inferencer._raw_data)

    self.assertEqual(
      [
       [2.0, 'y', datetime.datetime(2001, 2, 2, 0, 0), '1234-56-79', 2.3],
       [2.0, 'z', datetime.datetime(2001, 2, 2, 0, 0), '1234-56-79', 2.3]
      ],
      self._csv_schema_inferencer._raw_train_data)

    self.assertEqual(
      [
       [1.0, 'x', datetime.datetime(2000, 1, 1, 0, 0), '1234-56-78', 2.3],
      ],
      self._csv_schema_inferencer._raw_test_data)

    self.assertEqual(
      [[1.0], [0.0]],
      self._csv_schema_inferencer._raw_target_data)

  def testGetShuffledHeader(self):
    self.assertEqual(
      ['col1', 'col4', 'col6', 'col2_0', 'col2_1', 'col5_0'],
      self._csv_schema_inferencer._shuffled_header)

  def testGetData(self):
    data = self._csv_schema_inferencer.GetData()

    np.testing.assert_array_equal(
      np.array([
        [2.0, datetime.datetime(2001, 2, 2, 0, 0), 2.3, 1.0, 0.0, 1.0],
        [2.0, datetime.datetime(2001, 2, 2, 0, 0), 2.3, 0.0, 1.0, 1.0]]),
      data['X'])

    np.testing.assert_array_equal(
      np.array([[ 1.], [ 0.]]),
      data['y'])

    self.assertEqual(
      ['col1', 'col4', 'col6', 'col2_0', 'col2_1', 'col5_0'], 
      data['X_schema'])

    self.assertEqual('col3', data['y_schema'])

if __name__ == '__main__':
  unittest.main()
