from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import random
from datetime import datetime
 
def read(instance_file):
    infile = open(instance_file, 'r')
    Name = infile.readline().strip().split()[2]  # NAME
    infile.readline().strip().split()[2]         # TYPE
    infile.readline().strip().split()[2]         # COMMENT
    Dimension = infile.readline().strip().split()[2]  # DIMENSION
    infile.readline().strip().split()[2]         # EDGE_WEIGHT_TYPE
    infile.readline()
    print("Instance:", Name)
    print("Dimension:", Dimension)
    N = int(Dimension)
    nodelist = np.empty((N, 2))
    for i in range(N):
        x, y = infile.readline().strip().split()[1:]
        nodelist[i] = [x, y]
    infile.close()
    return nodelist
 
def calculate_path_length(path, distances):
    path_length = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        path_length += distances[current_city][next_city]
    return path_length
 
def cost_heursitic(path, distances):
    return 1.0 / calculate_path_length(path, distances)
 
def calculate_edge_desirability(current_city, next_city, pheromones, q_table, distances, delta, beta, mu, mode=0):
    phermone_comp = pheromones[current_city][next_city] ** delta
    distance_comp = cost_heursitic([current_city, next_city], distances) ** beta
    qlearning_comp = q_table[current_city][next_city] ** mu
    if mode == 0:
        return phermone_comp * distance_comp * qlearning_comp
    elif mode == 1:
        return phermone_comp * distance_comp
    elif mode == 2:
        return qlearning_comp
 
def choose_next_city(current_city, allowed_cities, q_table, pheromones, epsilon, distances, delta, beta, mu):
    desirabilities = [calculate_edge_desirability(current_city, next_city, pheromones, q_table, distances, delta, beta, mu, 0)
                      for next_city in allowed_cities]
    if random.random() < epsilon:
        desirabilities = np.array(desirabilities)
        desirabilities /= desirabilities.sum()
        return np.random.choice(allowed_cities, p=desirabilities)
    else:
        return allowed_cities[np.argmax(desirabilities)]
 
def update_q_table(q_table, s, a, alpha, distances, gamma):
    rewards = 1 / distances[s][a]
    q_table[s, a] += alpha * (rewards + gamma * np.max(q_table[a, :]) - q_table[s, a])
 
def update_pheromones(pheromones, paths, distances, w_reward, rho, num_cities):
    delta_pheromones = np.zeros((num_cities, num_cities))
    for path in paths:
        rewards = w_reward / calculate_path_length(path, distances)
        for i in range(len(path) - 1):
            current_city = path[i]
            next_city = path[i + 1]
            delta_pheromones[current_city][next_city] += rewards
    pheromones += -rho * pheromones + delta_pheromones
 
def local_search_2opt(path, distances, num_cities):
    best_path = path
    best_path_length = calculate_path_length(path, distances)
    improved = True
    while improved:
        improved = False
        for i in range(1, num_cities - 2):
            for j in range(i + 1, num_cities):
                if j - i == 1:
                    continue  
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_path_length = calculate_path_length(new_path, distances)
                if new_path_length < best_path_length:
                    best_path = new_path
                    best_path_length = new_path_length
                    improved = True
        path = best_path
    return best_path
 
def hybrid_qaco(local_alpha, local_epsilon, num_cities, distances, num_episodes, num_ants, num_local_search_iterations, 
                 gamma, alpha_decay_rate, epsilon_min, epsilon_decay_rate, w_reward, rho, delta, beta, mu):
    q_table = np.ones((num_cities, num_cities))
    pheromones = np.ones((num_cities, num_cities))
    for episode in range(num_episodes):
        paths = [[] for _ in range(num_ants)]
        path_lengths = np.zeros(num_ants)
        for hop in range(num_cities):
            for ant in range(num_ants):
                if hop == 0:
                    paths[ant].append(random.randint(0, num_cities - 1))
                allowed_cities = list(set(range(num_cities)) - set(paths[ant]))
                current_city = paths[ant][-1]
                if hop == num_cities - 1:
                    next_city = paths[ant][0]  # complete cycle
                else:
                    next_city = choose_next_city(current_city, allowed_cities, q_table, pheromones, local_epsilon,
                                                  distances, delta, beta, mu)
                paths[ant].append(next_city)
                path_lengths[ant] += distances[current_city][next_city]
            for ant in range(num_ants):
                update_q_table(q_table, paths[ant][-2], paths[ant][-1], local_alpha, distances, gamma)
        best_path_index = np.argmin(path_lengths)
        best_path = paths[best_path_index]
        best_path_length = path_lengths[best_path_index]
        for _ in range(num_local_search_iterations):
            new_path = local_search_2opt(best_path, distances, num_cities)
            new_path_length = calculate_path_length(new_path, distances)
            if new_path_length < best_path_length:
                best_path = new_path
                best_path_length = new_path_length
        update_pheromones(pheromones, [best_path], distances, w_reward, rho, num_cities)
        local_alpha *= 1.0 / (1.0 + alpha_decay_rate * episode)
        local_epsilon = max(epsilon_min, local_epsilon - epsilon_decay_rate)
    return best_path
 
# Global parameters (packaged into a dictionary for easy passing)
global_params = {
    "alpha": 0.5,                # Learning rate
    "alpha_decay_rate": 0.005,
    "gamma": 0.95,               # Discount factor
    "epsilon": 1.0,              # Exploration rate
    "epsilon_min": 0.005,
    "epsilon_decay_rate": 0.01,
    "num_ants": 20,
    "num_episodes": 100,
    "rho": 0.3,                  # Evaporation rate
    "w_reward": 10,
    "num_local_search_iterations": 25,
    "mu": 2,                     # Q-learning factor
    "delta": 2,                  # Pheromone factor
    "beta": 3                    # Heuristic factor
}
 
def run_trial(trial, num_cities, distances, global_params):
    local_alpha = global_params["alpha"]
    local_epsilon = global_params["epsilon"]
    best_path = hybrid_qaco(local_alpha, local_epsilon, num_cities, distances, 
                             global_params["num_episodes"], global_params["num_ants"],
                             global_params["num_local_search_iterations"], global_params["gamma"],
                             global_params["alpha_decay_rate"], global_params["epsilon_min"],
                             global_params["epsilon_decay_rate"], global_params["w_reward"],
                             global_params["rho"], global_params["delta"], global_params["beta"],
                             global_params["mu"])
    length = calculate_path_length(best_path, distances)
    return trial, best_path, length
 
#instances_dir = 'SmallInstances'
instances_dir = 'testInstances'
instance_files = [os.path.join(instances_dir, f) for f in os.listdir(instances_dir) if f.endswith('.tsp')]
 
for instance_file in instance_files:
    print("\n" + "="*60)
    print("Processing instance:", instance_file)
    cities = read(instance_file)
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            x1, y1 = cities[i]
            x2, y2 = cities[j]
            distances[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    num_trials = 5
    results = []
    start_time = datetime.now()
    print(f"\nStart Time: {start_time}\n{'='*40}")
    with ProcessPoolExecutor(max_workers=num_trials) as executor:
        futures = [executor.submit(run_trial, trial, num_cities, distances, global_params)
                   for trial in range(1, num_trials+1)]
        for future in as_completed(futures):
            trial, path, length = future.result()
            results.append((trial, path, length))
    results.sort(key=lambda x: x[0])
    for trial, path, length in results:
        print(f"\nTrial {trial}:\nOptimal Path Length: {length}\n{'-'*40}")
    avg_length = np.mean([length for _, _, length in results])
    print(f"\nAverage Optimal Path Length: {avg_length}")
    end_time = datetime.now()
    print(f"\nEnd Time: {end_time}")
    total_time = (end_time - start_time).total_seconds()
    print(f"Total Time Taken for instance: {total_time} seconds")