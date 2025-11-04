import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
import os

from tsp import extract_coords, nearest_neighbor, route_cost, create_distance_matrix

def unit_square_points(total_points, instances):
    os.makedirs('Unit-square-instances', exist_ok=True)  

    random.seed(None)

    edges = ['top', 'bottom', 'left', 'right']


    for i in range(instances):
        points = [(0.0,0.0), (1.0,0.0), (0.0,1.0), (1.0,1.0)]
        while len(points) < total_points:
            location = random.choice(edges)

            if location == 'top':
                x, y = random.random(), 1.0
            elif location == 'bottom':
                x, y = random.random(), 0.0
            elif location == 'left':
                x, y = 0.0, random.random()
            else:
                x, y = 1.0, random.random()

            if (x,y) not in points:
                points.append((round(x,7), round(y,7)))

        data = pd.DataFrame(points, columns=['x', 'y'])
        data.to_csv(f'Unit-square-instances/instance{total_points}',float_format='%.7e', sep=' ', index=False, header=None)

    return data


def plot_graph():
    trial_time = 5  
    foldername = 'Unit-square-instances'
    results_by_nodes = {}

    for entry in os.scandir(foldername):
        print(f"Running on: {entry.name}")
        data = extract_coords(os.path.join(foldername, entry.name))
        dist = create_distance_matrix(data)
        
        num_nodes = dist.shape[0]
        
        if num_nodes not in results_by_nodes:
            results_by_nodes[num_nodes] = {'random': [], 'nn': []}

        random_runs = []
        nn_runs = []

        for run in range(1): 
            random_cost = random_search_timed(dist, trial_time)
            nn_cost = anytime_nearest_neighbor_timed(dist, trial_time)
            random_runs.append(random_cost)
            nn_runs.append(nn_cost)

        avg_random = sum(random_runs) / len(random_runs)
        avg_nn = sum(nn_runs) / len(nn_runs)
        
        results_by_nodes[num_nodes]['random'].append(avg_random)
        results_by_nodes[num_nodes]['nn'].append(avg_nn)

    node_counts = sorted(results_by_nodes.keys())
    avg_random_costs = []
    avg_nn_costs = []
    
    for nodes in node_counts:
        avg_random = sum(results_by_nodes[nodes]['random']) / len(results_by_nodes[nodes]['random'])
        avg_nn = sum(results_by_nodes[nodes]['nn']) / len(results_by_nodes[nodes]['nn'])
        avg_random= (avg_random/4.0)*100.0
        avg_nn= (avg_nn/4.0)*100.0
        avg_random_costs.append(avg_random)
        avg_nn_costs.append(avg_nn)

    print("Node Counts:", node_counts)
    print("Average Random Search Costs:", avg_random_costs)
    print("Average Nearest Neighbor Costs:", avg_nn_costs)
    plt.scatter(node_counts, avg_nn_costs, color='blue', s=150, label='Algorithm A: Random')
    plt.scatter(node_counts, avg_random_costs, color='red', s=150, label='Algorithm B: Nearest Neighbor')
    plt.xscale('log', base=2)
    plt.xlim(20, 1100)
    plt.yticks(range(100, 17500, 2400))
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Cost % of Optimal')
    plt.legend()
    plt.title('Algorithm Comparison with 5 Second Time Budget')
    plt.axis('equal')
    plt.savefig('algorithm_comparison_5.png')


def generate_test_files():
    random.seed(None)

    #comment out error handlling in extract_coords func if size>256
    sizes = [512,1024]

    for size in sizes:
        points = set()
        while len(points) < size:
            x = round(random.uniform(0, 100), 7)
            y = round(random.uniform(0, 100), 7)
            points.add((x, y))

        data = pd.DataFrame(points, columns=['x', 'y'])
        data.to_csv(f'TSP_test_cases/instance_{size}_nodes.txt', float_format='%.7e', sep=' ', index=False, header=None)

    
def random_search_test(dist, time_limit, trials):
    total_distances = []
    total_times = []
    max_length = 0

    for i in range(trials):
        n = dist.shape[0]
        stops = list(range(2, n + 1))  

        best_cost = float('inf')
        best_route = None
        it = 0

        print(f"There are {n} nodes, computing route...")
        print("\n")
        print("Shortest Route Discovered So Far")

        distances = []
        times = []

        start_time = time.time()

        while time.time()-start_time < time_limit:
            random.shuffle(stops)
            route = [1] + stops + [1]
            cost = route_cost(route, dist)

            if cost < best_cost:
                best_cost = cost
                best_route = route[:]
                distances.append(best_cost)
                times.append((time.time()-start_time)/60.0)
                print(f"{best_cost:.1f}")

            it += 1

        if len(distances) > max_length:
            max_length = len(distances)

        total_distances.append(distances)
        total_times.append(times)

    for i, row in enumerate(total_distances):
        pad = row[-1]
        if len(row) < max_length:
            padding = max_length - len(row)
            total_distances[i] = row + [pad]*padding

    for i, row in enumerate(total_times):
        pad = row[-1]
        if len(row) < max_length:
            padding = max_length - len(row)
            total_times[i] = row + [pad]*padding

    avg_distance = np.vstack(total_distances)
    avg_time = np.vstack(total_times)

    x = np.mean(avg_time, axis=0)
    y = np.mean(avg_distance, axis=0)

    x.tofile('avg_time.txt', sep=', ')
    y.tofile('avg_dist.txt', sep=', ')

    plt.plot(x, y, label='Random Search')
    plt.title('')
    plt.xlabel('Time (in minutes)')
    plt.ylabel('Path Distance')
    plt.show()

    return best_cost, best_route

def random_search_timed(dist, trial_time):
    n = dist.shape[0]
    stops = list(range(2, n + 1))  

    best_cost = float('inf')
    it = 0

    start_time = time.time()

    while (time.time() < start_time + trial_time):
        random.shuffle(stops)
        route = [1] + stops + [1]
        cost = route_cost(route, dist)

        if cost < best_cost:
            best_cost = cost

        it += 1

    return best_cost


def anytime_nearest_neighbor_timed(dist, trial_time):
   n = dist.shape[0]
   best_so_far, best_route_so_far = nearest_neighbor(dist)

   it = 0
   
   start_time = time.time()
   
   while (time.time() < start_time + trial_time):
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
        it += 1

   return best_so_far


def unit_square_test(trial_time, foldername):
    random_costs = []
    nn_costs = []

    for entry in os.scandir(foldername):
        data = extract_coords(os.path.join(foldername, entry.name))
        dist = create_distance_matrix(data)
        print(entry.name)

        random_cost = random_search_timed(dist, trial_time)
        random_costs.append(random_cost)
        nn_cost = anytime_nearest_neighbor_timed(dist, trial_time)
        print(nn_cost)
        nn_costs.append(nn_cost)
    
    avg_random = sum(random_costs)/len(random_costs)
    avg_nn = sum(nn_costs)/len(nn_costs)

    print(f'Time: {trial_time} \n Random Search: {avg_random} \n Nearest Neighbor: {avg_nn}')

def main():
    #data = unit_square_points(1024, 1)
    plot_graph()

    # print('ComputeDronePath\n')

    # distance_matrix = create_distance_matrix(data)

    # best_cost, best_route = random_search_test(distance_matrix, 3, 3)
    # print(f"\nBest found: {best_cost:.1f}")
    # print(f"Route: {best_route}")

if __name__ == '__main__':
  main()