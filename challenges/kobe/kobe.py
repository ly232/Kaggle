##
# Following is required if $PYTHONPATH does not contain kaggle path.
import sys
from os.path import expanduser
sys.path.insert(0, expanduser(".") + '/../..')
##

from algorithms.classifications import Classifiers
from utils.csv_parser import CsvParser

def main():
  print "== Kobe shot classficiation ==\n"

  parser = CsvParser('data.csv', 'shot_made_flag')
  data = parser.GetData()

  classifiers = Classifiers()
  classifiers.Run(data['X'], data['y'])

  print classifiers.GetReport()

if __name__ == "__main__":
  main()
 