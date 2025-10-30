import pandas as pd
import numpy as np
import math as math
import random
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
import os

# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['x', 'y']     # label columns
    data = pd.read_csv(filename, sep=r'\s+', names=data_names)

    if not all (data.dtypes == 'float64'):
        raise Exception("Locations are not in float64 format.")

    if len(data) > 256:
        raise Exception("Error: Locations exceeds 256.")
    
    return data

def unit_square_points(total):
    random.seed(None)

    points = []
    edges = ['top', 'bottom', 'left', 'right']

    for i in range(total):
        location = random.choice(edges)

        if location == 'top':
            x, y = random.random(), 1.0
        elif location == 'bottom':
            x, y = random.random(), 0.0
        elif location == 'left':
            x, y = 0.0, random.random()
        else:
            x, y = 1.0, random.random()

        points.append([x, y])
    
    data = pd.DataFrame(points, columns=[x, y])
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
    print("     Shortest Route Discovered So Far using Random Search")


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
                print(f"          {best_cost:.1f}")

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
   print("     Shortest Route Discovered So Far using Nearest Neighbor Heuristic")
   it = 0

   print(f"          {best_so_far:.1f}")

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
               print(f"          {best_so_far:.1f}")
           it += 1


   except KeyboardInterrupt:
       print("Stopped with Ctrl+C (forced exit)")


   return best_so_far, best_route_so_far


def plot_graph(best_route,data,file_name, distance,search):
    x = []
    y = []
    for loc in best_route:
        x.append(data.iloc[loc-1, 0])
        y.append(data.iloc[loc-1, 1])

    plt.plot(x, y, color = 'black', marker = 'o', markersize = 6, markerfacecolor = 'blue')
    plt.plot(x[0], y[0], marker = 'o', markersize = 13, markerfacecolor = 'red')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'Best So Far Route for TSP using {search}', fontsize=15)
    plt.savefig(f"{file_name}_solution_{distance}")
    plt.close()

    

def main():
    data = unit_square_points(128)

    print('\nComputeDronePath')


    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)

    distance_matrix = create_distance_matrix(data)


    best_cost, best_route = random_search(distance_matrix)
    distance = int(best_cost)
    print(f"D, A total distance for the route: {distance}")
    if distance > 6000: 
        print(f"Warning: Solution is {distance}, greater than the 6000-meter constraint." + "\n")
    print(f"Route written to disk as {file_name}_solution_{distance}.txt" + "\n" + "\n")
    with open(f"{file_name}_solution_{distance}.txt","w") as f:
        for i,point in enumerate(best_route):
            f.write(f"{point}")
            if i < len(best_route) - 1:
                f.write("\n")
    plot_graph(best_route,data,file_name,distance,"Random Search")

    total_cost, nearest_neighbor_route = anytime_nearest_neighbor(distance_matrix)
    distance = int(total_cost)
    print(f"D, A total distance for the route: {distance}")
    if distance > 6000: 
        print(f"\nWarning: Solution is {distance}, greater than the 6000-meter constraint." + "\n" )
    print(f"Route written to disk as {file_name}_solution_{distance}.txt"  + "\n" + "\n")
    with open(f"{file_name}_solution_{distance}.txt","w") as f:
        for i,point in enumerate(nearest_neighbor_route):
            f.write(f"{point}")
            if i < len(nearest_neighbor_route) - 1:
                f.write("\n")
    plot_graph(nearest_neighbor_route,data,file_name,distance,"Anytime NN Search")

if __name__ == '__main__':
  main()