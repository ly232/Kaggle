import histogram

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer_pd = pd.DataFrame(data= np.c_[cancer['data'], cancer['target']],
                         columns= np.append(cancer['feature_names'], 'target'))

histogram.MakeHistograms(cancer_pd, 'target', 15)
