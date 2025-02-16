import numpy as np
import random

def read():

    infile = open('tsplib/eil51.tsp', 'r')

    Name = infile.readline().strip().split()[2] # NAME
    infile.readline().strip().split()[2] # TYPE
    infile.readline().strip().split()[2] # COMMENT
    Dimension = infile.readline().strip().split()[2] # DIMENSION
    infile.readline().strip().split()[2] # EDGE_WEIGHT_TYPE
    infile.readline()
    print(Name)
    print(Dimension)

    N = int(Dimension)
    nodelist = np.empty(shape=(N,2))
    for i in range(0, N):
        x,y = infile.readline().strip().split()[1:]
        nodelist[i] = [x,y]

    infile.close()
    return nodelist
cities = read()

num_cities = len(cities)
distances = np.zeros((num_cities, num_cities))

# Calculate the Euclidean distance between each pair of cities
for i in range(num_cities):
    for j in range(num_cities):
        x1, y1 = cities[i]
        x2, y2 = cities[j]
        distances[i][j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Q-learning parameters
alpha = 0.3  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_iterations = 100
num_ants = 20

# Local search parameters
num_local_search_iterations = 20


def choose_next_city(current_city, allowed_cities, q_table):
    if random.random() < epsilon:
        # Explore
        return random.choice(allowed_cities)
    else:
        # Exploit
        q_values = q_table[current_city, allowed_cities]
        return allowed_cities[np.argmax(q_values)]

def update_q_table(q_table, path, rewards):
    for i in range(num_cities - 1):
        current_city = path[i]
        next_city = path[i + 1]
        # Compute the updated Q-value using the discount factor and the learning rate
        #q_table[current_city, next_city] += alpha * (rewards + gamma * np.max(q_table[next_city]) - q_table[current_city, next_city])
        q_table[current_city, next_city] += alpha * (rewards - q_table[current_city, next_city])

def calculate_path_length(path):
    path_length = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        path_length += distances[current_city][next_city]
    path_length += distances[path[-1]][path[0]]  # Complete the cycle
    return path_length

def update_pheromones(paths):
    global pheromones  # Add this line to reference the global variable
    delta_pheromones = np.zeros((num_cities, num_cities))

    for path in paths:
        path_length = len(path)
        rewards = 1.0 / path_length
        for i in range(path_length - 1):
            current_city = path[i]
            next_city = path[i + 1]
            delta_pheromones[current_city][next_city] += rewards

    pheromones = (1 - alpha) * pheromones + delta_pheromones

def local_search_2opt(path):
    best_path = path
    best_path_length = calculate_path_length(path)

    improved = True
    while improved:
        improved = False
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue  # Skip adjacent cities
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_path_length = calculate_path_length(new_path)
                if new_path_length < best_path_length:
                    best_path = new_path
                    best_path_length = new_path_length
                    improved = True
        path = best_path

    return best_path


# Q-learning with ant colony optimization
q_table = np.zeros((num_cities, num_cities))  
pheromones = np.ones((num_cities, num_cities))

for iteration in range(num_iterations):
    best_path_length = float('inf')
    best_path = []

    # Generate paths for each ant
    for ant in range(num_ants):
        current_city = random.randint(0, num_cities - 1)
        visited_cities = set([current_city])
        path = [current_city]
        path_length = 0

        while len(visited_cities) < num_cities:
            allowed_cities = list(set(range(num_cities)) - visited_cities)
            next_city = choose_next_city(current_city, allowed_cities, q_table)
            visited_cities.add(next_city)
            path.append(next_city)
            path_length += distances[current_city][next_city]
            current_city = next_city

        path_length += distances[path[-1]][path[0]]  # Complete the cycle
        if path_length < best_path_length:
            best_path_length = path_length
            best_path = path

        update_q_table(q_table, path, 1.0 / path_length)
    update_pheromones([best_path])

    # Apply 2-opt local search to the best path
    best_path = local_search_2opt(best_path)
    best_path_length = calculate_path_length(best_path)

    # Perform additional iterations of local search
    for _ in range(num_local_search_iterations):
        new_path = local_search_2opt(best_path)
        new_path_length = calculate_path_length(new_path)
        if new_path_length < best_path_length:
            best_path = new_path
            best_path_length = new_path_length

    # Update Q-table using an incremental update rule
    for i in range(num_cities - 1):
        current_city = best_path[i]
        next_city = best_path[i + 1]
        q_table[current_city, next_city] += alpha * (1.0 / best_path_length - q_table[current_city, next_city])

# Find the optimal solution
optimal_path = best_path
optimal_path_length = best_path_length

print("Optimal Path:", optimal_path)
print("Optimal Path Length:", optimal_path_length)