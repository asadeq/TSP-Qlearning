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

def calculate_path_length(path):
    path_length = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        path_length += distances[current_city][next_city]
    return path_length

def cost_heursitic(path):
    return 1.0 / calculate_path_length(path)

def calculate_edge_desirability(current_city, next_city, pheromones, q_table):
    phermone_comp = pheromones[current_city][next_city]**delta
    distance_comp = cost_heursitic([current_city, next_city])**beta
    qlearning_comp = q_table[current_city][next_city]**mu
    return phermone_comp * distance_comp * qlearning_comp

def choose_next_city(current_city, allowed_cities, q_table, pheromones, epsilon):
    edge_desirabilities = [calculate_edge_desirability(current_city, next_city, pheromones, q_table) for next_city in allowed_cities]
    if random.random() < epsilon:
        # Explore (random)
        #return random.choice(allowed_cities)
        # Explore weighted by edge desirability        
        edge_desirabilities = np.array(edge_desirabilities)
        edge_desirabilities /= edge_desirabilities.sum()
        return np.random.choice(allowed_cities, p=edge_desirabilities)
    else:
        # Exploit by highest edge desirability
        return allowed_cities[np.argmax(edge_desirabilities)]

# Compute the updated Q-value using the discount factor and the learning rate
def update_q_table(q_table, s, a, alpha):
    rewards =  1 / distances[s][a]
    q_table[s, a] += alpha * (rewards + gamma * np.max(q_table[a]) - q_table[s, a])

def update_pheromones(pheromones, paths):
    delta_pheromones = np.zeros((num_cities, num_cities))
    for path in paths:
        # deposit pheromones on each edge of the best path
        rewards = w_reward / calculate_path_length(path)
        for i in range(len(path) - 1):
            current_city = path[i]
            next_city = path[i + 1]
            delta_pheromones[current_city][next_city] += rewards

    # Update the original pheromones using the evaporation rate and the new pheromones
    pheromones += -rho * pheromones + delta_pheromones
    

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

def hybrid_qaco(alpha, epsilon):

    q_table = np.ones((num_cities, num_cities))  
    pheromones = np.ones((num_cities, num_cities))

    for iteration in range(num_iterations):
        best_path_length = float('inf')
        best_path = []

        # initalize ant paths
        paths = [[] for _ in range(num_ants)]        
        path_lengths = np.zeros(num_ants)

        for hop in range(num_cities):

            for ant in range(num_ants):
                # Start a new path from a random city
                if hop == 0:
                    paths[ant].append(random.randint(0, num_cities - 1))            

                # Choose next city
                allowed_cities = list(set(range(num_cities)) - set(paths[ant]))
                current_city = paths[ant][-1]
                if hop == num_cities - 1:
                    next_city = paths[ant][0]  # Complete the cycle
                else:
                    next_city = choose_next_city(current_city, allowed_cities, q_table, pheromones, epsilon)

                # update path
                paths[ant].append(next_city)
                path_lengths[ant] += distances[current_city][next_city]

            # Update Q-table for each ant after the hop
            for ant in range(num_ants):
                update_q_table(q_table, paths[ant][-2], paths[ant][-1], alpha)
        
        # Get the best path of this iteration
        best_path_index = np.argmin(path_lengths)
        best_path = paths[best_path_index]
        best_path_length = path_lengths[best_path_index]

        # Apply 2-opt local search to the best path
        for _ in range(num_local_search_iterations):
            new_path = local_search_2opt(best_path)
            new_path_length = calculate_path_length(new_path)
            if new_path_length < best_path_length:
                best_path = new_path
                best_path_length = new_path_length
        
        # Update pheromones using the best path found this iteration
        update_pheromones(pheromones, [best_path])

        # Decay Q-learning rate
        alpha *= 1.0 / (1.0 + alpha_decay_rate * iteration)
        epsilon *= 1.0 / (1.0 + epsilon_decay_rate * iteration)

        # Update Q-table using an incremental update rule
        # for i in range(num_cities - 1):
        #     current_city = best_path[i]
        #     next_city = best_path[i + 1]
        #     q_table[current_city, next_city] += alpha * (1.0 / best_path_length - q_table[current_city, next_city])

    return best_path


#####################################################################################################

# Q-learning parameters
alpha = 0.5  # Learning rate
alpha_decay_rate = 0.005  # Learning rate decay
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.05  # Minimum exploration rate
epsilon_decay_rate = 0.01  # Exploration rate decay

# Ant colony optimization parameters
num_ants = 20
num_iterations = 100
rho = 0.1  # Evaporation rate
w_reward = 2

# Local search parameters (for 2-opt)
num_local_search_iterations = 2        #2

# Q-learning with ant colony optimization
mu = 1 # Q-learning factor
delta = 1  # Pheromone factor
beta = 1 # Heuristic factor

num_trials = 2
optimal_paths = []
optimal_path_lengths = []
for trial in range(num_trials):
    optimal_paths.append(hybrid_qaco(alpha, epsilon))
    optimal_path_lengths.append(calculate_path_length(optimal_paths[trial]))
    
    # Print trial number and optimal path length
    print("Trial:", trial + 1)
    print("Optimal Path Length:", optimal_path_lengths[trial])

# print average of optimal path lengths
print("Average Optimal Path Length:", np.mean(optimal_path_lengths))

