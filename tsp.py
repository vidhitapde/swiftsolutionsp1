import numpy as np
import pandas as pd
import math as math
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import threading

# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['x', 'y']     # label columns
    data = pd.read_csv(filename, sep='\s+', names=data_names)
    return data

# calculate euclidean distance to be used in creating distance matrix
def euclidean_distance(point1, point2):
    return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)

def create_distance_matrix(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))   #N x N matrix of zeros
   
    for i in range(n):
        for j in range(i, n):
            dist_matrix[i, j] = euclidean_distance(data.iloc[i], data.iloc[j])

    return dist_matrix


def route_cost(route_1_based, dist):
    cost = 0.0
    for k in range(len(route_1_based) - 1):
        a = route_1_based[k]   - 1  
        b = route_1_based[k+1] - 1
        cost += dist[a, b]
    return cost

def random_search(dist):
    n = dist.shape[0]
    stops = list(range(2, n + 1))  

    best_cost = float('inf')
    best_route = None
    it = 0

    print(f"There are {n} nodes, computing route...")
    print("\n")
    print("Shortest Route Discovered So Far")


    stop_flag = threading.Event()

    def wait_for_enter():
        try:
            input()            
            stop_flag.set()
        except EOFError:
            pass

    threading.Thread(target=wait_for_enter, daemon=True).start()

    try:
        while not stop_flag.is_set():
            random.shuffle(stops)
            route = [1] + stops + [1]
            cost = route_cost(route, dist)

            if cost < best_cost:
                best_cost = cost
                best_route = route[:]
                print(f"{best_cost:.1f}")

            it += 1
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C (forced exit)")


    return best_cost, best_route

def plot_graph(dist,best_route,data):
    x = []
    y = []
    for loc in best_route:
        x.append(data.iloc[loc-1, 0])
        y.append(data.iloc[loc-1, 1])

    plt.plot(x, y, color = 'black', marker = 'o', markersize = 6, markerfacecolor = 'red')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Best So Far Route for TSP')
    plt.show()


def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    data = extract_coords(filename)

    distance_matrix = create_distance_matrix(data)

    best_cost, best_route = random_search(distance_matrix)
    print(f"\nBest found: {best_cost:.1f}")
    print(f"Route: {best_route}")

    plot_graph(distance_matrix,best_route,data)

if __name__ == '__main__':
  main()