import numpy as np
import pandas as pd
import math as math

def extract_coords(filename):
    data_names = ['x', 'y']
    data = pd.read_csv(filename, sep='\s+', skip_blank_lines=True)
    data.info(verbose=True)
    return data

def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    data = extract_coords(filename)

if __name__ == '__main__':
  main()