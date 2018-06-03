import numpy as np
import unittest

from csv_parser import CsvParser

class TestSchemaInferencer(unittest.TestCase):
  def setUp(self):
    self._csv_parser = CsvParser(
      'testdata/csv_parser_test.csv', 'col3')

  def testGetRawData(self):
    self.assertEqual(
      ['col' + str(i + 1) for i in range(6)], 
      self._csv_parser._raw_header)

    self.assertEqual(
      [
       [1.0, 'y', '', 946713600.0, '1234-56-79', 2.3],
       [2.0, 'y', 1.0, 981100800.0, '1234-56-79', 2.3],
       [2.0, 'z', 0.0, 981100800.0, '1234-56-79', 2.3]
      ],
      self._csv_parser._raw_data)

    self.assertEqual(
      [
       [2.0, 'y', 981100800.0, '1234-56-79', 2.3],
       [2.0, 'z', 981100800.0, '1234-56-79', 2.3]
      ],
      self._csv_parser._raw_train_data)

    self.assertEqual(
      [
       [1.0, 'y', 946713600.0, '1234-56-79', 2.3],
      ],
      self._csv_parser._raw_test_data)

    self.assertEqual(
      [1.0, 0.0],
      self._csv_parser._raw_target_data)

  def testGetShuffledHeader(self):
    self.assertEqual(
      ['col1', 'col4', 'col6', 'col2_0', 'col2_1', 'col5_0'],
      self._csv_parser._shuffled_header)

  def testGetData(self):
    data = self._csv_parser.GetData()

    np.testing.assert_array_equal(
      np.array([
        [2.0, 9.811008e+08, 2.3, 1.0, 0.0, 1.0],
        [2.0, 9.811008e+08, 2.3, 0.0, 1.0, 1.0]]),
      data['X'])

    np.testing.assert_array_equal(
      np.array([
        [1.0, 9.467136e+08, 2.3, 1.0, 0.0, 1.0]]),
      data['X_test'])

    np.testing.assert_array_equal(
      np.array([1., 0.]),
      data['y'])

    self.assertEqual(
      ['col1', 'col4', 'col6', 'col2_0', 'col2_1', 'col5_0'], 
      data['X_schema'])

    self.assertEqual('col3', data['y_schema'])

if __name__ == '__main__':
  unittest.main()
