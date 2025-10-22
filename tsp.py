import numpy as np
import pandas as pd

def extract_coords(filename):
    data = pd.read_csv(filename)
    data.head()
    print(data)

def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    extract_coords(filename)

if __name__ == '__main__':
  main()