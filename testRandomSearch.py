import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time

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

    print(y)

    plt.plot(x, y, label='Random Search')
    plt.title('')
    plt.xlabel('Time (in minutes)')
    plt.ylabel('Path Distance')
    plt.show()

    return best_cost, best_route

def main():
    data = unit_square_points(128)

    print('ComputeDronePath\n')

    distance_matrix = create_distance_matrix(data)

    best_cost, best_route = random_search_test(distance_matrix, 3, 3)
    print(f"\nBest found: {best_cost:.1f}")
    print(f"Route: {best_route}")

if __name__ == '__main__':
  main()