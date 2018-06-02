import numpy as np

_RAW_FILE = 'data_small.csv'

def LoadData(filepath):
  res = np.loadtxt(filepath, dtype=str, delimiter=',', skiprows=1)
  return res

def main():
  print "== Kobe shot classficiation =="
  data = LoadData(_RAW_FILE)
  print data

if __name__ == "__main__":
  main()
 