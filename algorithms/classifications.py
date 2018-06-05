"""Convinient wrapper for various off-the-shelf popular ML classifiers.

This file is effectively a configuration file, specifying (a) the classifiers
to be considered, and (b) the hyper-parameters for each classifier. It makes no
attempt to do any kind of optimizations what-so-ever.
"""

import threading

from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import discriminant_analysis 

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""Classifier configurations.

Keys are the classifier class names, value is a
pair consisting of the sklearn package name and the model constructor 
parameters.
"""
_CLASSIFIER_CONFIGS = {
  'LogisticRegression': (linear_model, {'C': 50}),
  'LinearSVC': (svm, {'C': 50}),
  'DecisionTreeClassifier': (tree, {}),
  'AdaBoostClassifier': (ensemble, {}),
  'RandomForestClassifier': (ensemble, {'n_estimators': 50}),
  'GradientBoostingClassifier': (ensemble, {}),
  # 'MLPClassifier': (neural_network, {}),
  # 'GaussianProcessClassifier': (gaussian_process, {}),
  # 'GaussianNB': (naive_bayes, {}),
  # 'QuadraticDiscriminantAnalysis': (discriminant_analysis, {}),
}


"""Thread to execute ML classification algorithm.
"""
class RunnerThread(threading.Thread):
  def __init__(self, model, X, y, score):
    threading.Thread.__init__(self)
    self._model = model
    self._X = X
    self._y = y
    self._score = score

  def run(self):
    X_train, X_test, y_train, y_test = train_test_split(self._X, self._y)
    clf = self._model.fit(X_train, y_train)
    self._score['training'] = clf.score(X_train, y_train)
    self._score['testing'] = clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    self._score['report'] = classification_report(y_test, predicted)


"""Class that holds various classifiers.

It kicks start ML classification runs in parallel, and generates final report.
"""
class Classifiers(object):

  def __init__(self):
    self._models = {
      model: getattr(
        _CLASSIFIER_CONFIGS[model][0], model)(**_CLASSIFIER_CONFIGS[model][1])
      for model in _CLASSIFIER_CONFIGS
    }

    self._scores = {
      model: {
        'training': 0,
        'testing': 0,
        'report': 'N.A.',
      }
      for model in _CLASSIFIER_CONFIGS
    }

  def Run(self, X, y):
    threads = []
    for model in self._models:
      thread = RunnerThread(
        self._models[model], 
        StandardScaler().fit_transform(X), 
        y, 
        self._scores[model])
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

  def Predict(self, model, X):
    return self._models[model].predict(X)

  def GetReport(self):
    report = '== Classifiers Comparison Report ==\n\n'
    for model in self._scores:
      score = self._scores[model]
      report += model + '\n'
      report += '  Training set score: {:.3f}\n'.format(score['training'])
      report += '  Testing set score: {:.3f}\n'.format(score['testing'])
      report += score['report']
      report += '\n\n'
    return report
