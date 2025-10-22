import numpy as np
import pandas as pd
import math as math

def extract_coords(filename):
    data_names = ['x', 'y']
    data = pd.read_csv(filename, sep='\s+', names=data_names)
    print(data)
    data.info(verbose=True)
    return data

def euclidean_distance(point1, point2):
   return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)

def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    data = extract_coords(filename)

    print(euclidean_distance(data.iloc[0], data.iloc[1]))

if __name__ == '__main__':
  main()