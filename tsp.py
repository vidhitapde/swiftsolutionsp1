import numpy as np
import pandas as pd
import math as math

def extract_coords(filename):
    data_names = ['x', 'y']
    data = pd.read_csv(filename, sep='\s+', names=data_names)
    return data

def euclidean_distance(point1, point2):
    return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)

def create_distance_matrix(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))   #N x N matrix of zeros
   
    for i in range(n):
        for j in range(i, n):
            dist_matrix[i, j] = euclidean_distance(data.iloc[i], data.iloc[j])

def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    data = extract_coords(filename)

    distance_matrix = create_distance_matrix(data)

if __name__ == '__main__':
  main()