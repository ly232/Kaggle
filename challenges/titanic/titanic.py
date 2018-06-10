##
# Following is required if $PYTHONPATH does not contain kaggle path.
import sys
from os.path import expanduser
sys.path.insert(0, expanduser(".") + '/../..')
##

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.pandas_utils as pd_utils
import algorithms.classifications as clfs


def CleanupData(samples):
  """Cleans up the raw data frame loaded from CSV.

  Args:
    df (pd.DataFrame): Raw data frame to clean up.

  Returns: New data frame. Rows may be dropped and colums may be pruned (e.g.
    for features irrelevant to prediction) or added (e.g. for 
    one-hot-encoding).
  """
  # Exclude columns that are either irrelevant to predictions, or are missing
  # in the test dataset.
  #
  # NOTE: 'Cabin' is an interesting one. It's a non-numerical column so it 
  # requires one hot encoding. However, the cardinality is quite large, e.g.
  # O(100). With one hot encoding it will drastically increase the 
  # dimensionality of the search space.
  samples = pd_utils.DropColumns(samples, ['Name', 'Ticket', 'Fare', 'Cabin'])

  # Apply one-hot-encoding to all non-numeric columns.
  samples = pd_utils.OneHotEncode(samples)

  # Drop any columns that may have null entries.
  # TODO: how can we handle null columns?
  samples = pd_utils.ProjectOutNullColumns(samples)

  return samples


def main():
  #
  # Data loading and cleanup.
  #
  training_data = pd.read_csv('train.csv')
  schema = training_data.columns
  samples = training_data[list(set(schema) - set(['Survived']))]
  targets = training_data['Survived']
  samples = CleanupData(samples)

  #
  # Generate predictions.
  #
  samples_test = pd.read_csv('test.csv')
  samples_test = CleanupData(samples_test)

  # samples may still have more columns that samples_test due to 
  # one-hot-encoding, so we need to further augment columns for samples_test.
  extra_cols = list(set(samples.columns) - set(samples_test))
  samples_test = pd_utils.AugmentColumns(
    samples_test, {col: 0.0 for col in extra_cols})

  # Drop columns, e.g. one-hot-encoded cabin columns in test data but not 
  # training data.
  drop_cols = list(set(samples_test.columns) - set(samples.columns))
  samples_test = pd_utils.DropColumns(samples_test, drop_cols)

  # Make predictions and write result to file.
  assert set(samples.columns) == set(samples_test.columns)

  classifiers = clfs.Classifiers()
  classifiers.Run(samples, targets)
  print classifiers.GetReport()
  predictions = classifiers.Predict('GradientBoostingClassifier', samples_test)

  # voting_classifier = clfs.GetVotingClassifier()
  # voting_classifier.fit(samples, targets)
  # predictions = voting_classifier.predict(samples_test)
  
  print "Predictions:\n{}".format(predictions)

  passenger_ids = samples_test['PassengerId']
  output = np.transpose(np.vstack((passenger_ids, predictions)))
  np.savetxt(
    'submission.csv',
    output,
    fmt='%d',
    delimiter=',',
    header='PassengerId,Survived',
    comments='')


if __name__ == '__main__':
  main()
 