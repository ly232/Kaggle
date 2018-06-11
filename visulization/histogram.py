r"""
Histogram related visulization utilities.
"""

import math
import matplotlib.pyplot as plt
import numpy as np

def MakeHistograms(data, target_colname, plot_rows):
  """Makes subplots of histograms for the given binary classifications data.
  TODO: support multi-class classifications.

  Args:
    data (pandas.DataFrame): Data points, including both samples and target.
    target_colname (str): Column name for the target column.
    plot_rows (int): Number of rows to appear in figure. 
  """
  n_samples, n_features = data.shape
  n_features -= 1  # Discount the target column.
  plot_columns = int(math.ceil(n_features / plot_rows))

  fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(10, 20))
  class_0 = data[data[target_colname] == 0]
  class_1 = data[data[target_colname] == 1]

  ax = axes.ravel()

  features = list(set(data.columns) - set([target_colname]))
  for i in range(len(features)):
    feature = features[i]
    print i, feature
    _, bins = np.histogram(data[feature], bins=50)
    ax[i].hist(class_0[feature].values, bins=bins, alpha=0.5)
    ax[i].hist(class_1[feature].values, bins=bins, alpha=0.5)
    ax[i].set_title(feature)
    ax[i].set_yticks(())

  ax[0].legend(['class 0', 'class 1'], loc='best')
  fig.tight_layout()
  plt.show()
