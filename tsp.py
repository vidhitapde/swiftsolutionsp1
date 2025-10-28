import pandas as pd
import numpy as np
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

def anytime_nearest_neighbor(dist):
   n = dist.shape[0]
   best_so_far, best_route_so_far = nearest_neighbor(dist)

   print(f"There are {n} nodes, computing route...")
   print("\n")
   print("Shortest Route Discovered with Nearest Neighbor Heuristic")
   it = 0

   print(f"{best_so_far:.1f}")

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
           remaining_locations = list(range(2, n + 1)) #starts from location 2, because location 1 is starting spot
           current_optimal_path = [1] #always start with the first x,y coordinate in txt file
           total_cost = 0.0
           current_location = 1
           while remaining_locations:
               best_cost = float('inf') #current distance will always be less than this, also need to reset for each loc
               second_best_cost = float('inf')
               nearby_loc = None
               second_nearby_loc = None
               for loc in remaining_locations:
                   #euclidian distance for (x1,y1) and (x2,y2) represented by point value in row and col in the dist matrix
                   current_distance = dist[current_location-1][loc-1]
                   if(current_distance < best_cost) and current_distance != 0.0:
                       second_best_cost = best_cost
                       best_cost = current_distance
                       second_nearby_loc = nearby_loc
                       nearby_loc = loc # need to put the path together
                   elif (current_distance < second_best_cost) and current_distance != 0.0:
                       second_best_cost = current_distance
                       second_nearby_loc = loc
               if(random.random() < 0.1  and second_nearby_loc in remaining_locations and second_nearby_loc is not None):
                   best_cost = second_best_cost
                   nearby_loc = second_nearby_loc
              
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
           if total_cost < best_so_far:
               best_so_far = total_cost
               best_route_so_far = current_optimal_path[:]
               print(f"{best_so_far:.1f}")
           it += 1


   except KeyboardInterrupt:
       print("Stopped with Ctrl+C (forced exit)")


   return best_so_far, best_route_so_far




  
def plot_graph(dist,best_route,data):
    x = []
    y = []
    for loc in best_route:
        x.append(data.iloc[loc-1, 0])
        y.append(data.iloc[loc-1, 1])

    plt.plot(x, y, color = 'black', marker = 'o', markersize = 6, markerfacecolor = 'blue')
    plt.plot(x[0], y[0], marker = 'o', markersize = 13, markerfacecolor = 'red')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Best So Far Route for TSP', fontsize=15)
    plt.show()

def main():
    print('ComputeDronePath\n')
    filename = input('Enter the name of file: ')
    data = extract_coords(filename)

    distance_matrix = create_distance_matrix(data)
    print("Distance Matrix:")
    print(distance_matrix)

    best_cost, best_route = random_search(distance_matrix)
    print(f"\nBest found: {best_cost:.1f}")
    print(f"Route: {best_route}")
    
    plot_graph(distance_matrix,best_route,data)


    # total_cost, nearest_neighbor_route = nearest_neighbor(distance_matrix)
    # print(f"\nNearest Neighbor Cost: {total_cost:.1f}")
    # print(f"Nearest Neighbor Route: {nearest_neighbor_route}")
    # plot_graph(distance_matrix,nearest_neighbor_route,data)

    total_cost, nearest_neighbor_route = anytime_nearest_neighbor(distance_matrix)
    print(f"\nAnytime Nearest Neighbor Cost: {total_cost:.1f}")
    print(f"Anytime Nearest Neighbor Route: {nearest_neighbor_route}")
    plot_graph(distance_matrix,nearest_neighbor_route,data)

if __name__ == '__main__':
  main()