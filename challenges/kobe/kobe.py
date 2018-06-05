##
# Following is required if $PYTHONPATH does not contain kaggle path.
import sys
from os.path import expanduser
sys.path.insert(0, expanduser(".") + '/../..')
##

import numpy as np

from algorithms.classifications import Classifiers
from utils.csv_parser import CsvParser

def main():
  parser = CsvParser('data.csv', 'shot_made_flag')
  data = parser.GetData()

  shot_id_index = data['X_schema'].index('shot_id')
  test_shot_ids = data['X_test'][:, shot_id_index]

  classifiers = Classifiers()
  classifiers.Run(data['X'], data['y'])
  print classifiers.GetReport()

  predictions = classifiers.Predict('LinearSVC', data['X_test'])
  output = np.transpose(np.vstack((test_shot_ids, predictions)))
  np.savetxt(
    "submission.csv",
    output,
    fmt='%d',
    delimiter=',',
    header='shot_id,shot_made_flag',
    comments='')

if __name__ == '__main__':
  main()
 