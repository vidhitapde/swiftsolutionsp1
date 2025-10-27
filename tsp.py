import numpy as np
import pandas as pd
import math as math
import random
import sys
from pathlib import Path
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
            dist_matrix[i, j] = dist_matrix[j, i] = euclidean_distance(data.iloc[i], data.iloc[j])

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


def nearest_neighbor(dist):
    n = dist.shape[0]
    remaining_locations = list(range(2, n + 1)) #starts from location 2, because location 1 is starting spot

    best_cost = float('inf')
    best_route = None
    current_optimal_path = [1] #always start with the first x,y coordinate in txt file
    total_cost = 0.0
    current_location = 1
    print(f"There are {n} nodes, computing route...")
    print("\n")
    print("Shortest Route Discovered with Nearest Neighbor Heuristic")

    while remaining_locations:
        best_cost = float('inf') #current distance will always be less than this, also need to reset for each loc
        for loc in remaining_locations:
            #euclidian distance for (x1,y1) and (x2,y2) represented by point value in row and col in the dist matrix
            current_distance = dist[current_location-1][loc-1]
            if(current_distance == 0):
                current_distance = dist[loc-1][current_location-1] #swap to get distance at point, currently 0.0 as already calculated
            #check for the nearby location to the current_location
            if(current_distance < best_cost) and current_distance != 0.0:
                best_cost = current_distance
                nearby_loc = loc # need to put the path together
        total_cost = total_cost + best_cost #putting together the total distance for the route
        current_optimal_path.append(nearby_loc)
        current_location = nearby_loc
        remaining_locations.remove(nearby_loc)

    remaining_locations.append(1) #appends start location to get to the end of the route from that point
    if(len(remaining_locations) == 1):
        current_distance = dist[remaining_locations[0]-1][current_optimal_path[-1]-1]
        total_cost += current_distance
        current_optimal_path.append(remaining_locations[0])
        remaining_locations.remove(remaining_locations[0])

    return total_cost, current_optimal_path


def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    data = extract_coords(filename)

    distance_matrix = create_distance_matrix(data)

    best_cost, best_route = random_search(distance_matrix)
    print(f"\nBest found: {best_cost:.1f}")
    print(f"Route: {best_route}")

    total_cost, nearest_neighbor_route = nearest_neighbor(distance_matrix)
    print(f"\nNearest Neighbor Cost: {total_cost:.1f}")
    print(f"Nearest Neighbor Route: {nearest_neighbor_route}")

if __name__ == '__main__':
  main()