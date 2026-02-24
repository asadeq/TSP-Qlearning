import math
import time
import numpy as np
import random

class AntQ:
    def __init__(self, distance_matrix, m=None, alpha=0.1, gamma=0.3, delta=1, beta=2, q0=0.9, W=10):
        """
        Initializes the Ant-Q algorithm parameters.
        """
        self.distances = np.array(distance_matrix, dtype=float)
        np.fill_diagonal(self.distances, np.inf) # Prevent self-loops
        self.n = len(self.distances)
        
        # Number of agents: defaults to number of cities if not specified 
        self.m = m if m is not None else self.n
        
        # Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.beta = beta
        self.q0 = q0
        self.W = W
        
        # Heuristic values HE(r, s) = 1 / d_{rs}
        with np.errstate(divide='ignore'):
            self.HE = 1.0 / self.distances
        
        # Initialize AQ values (AQ_0 = 1 / (avg_edge_length * n)) 
        avg_length = np.mean(self.distances[self.distances != np.inf])
        self.AQ0 = 1.0 / (avg_length * self.n)
        self.AQ = np.full((self.n, self.n), self.AQ0)
        
    def _action_choice(self, r, unvisited):
        """
        Pseudo-random-proportional action choice rule.
        """
        q = random.random()
        
        # Calculate the heuristic and learned values for available moves
        aq_vals = self.AQ[r, unvisited]
        he_vals = self.HE[r, unvisited]
        evaluations = (aq_vals ** self.delta) * (he_vals ** self.beta) 
        
        if q <= self.q0:
            # Exploitation: Choose the best available edge deterministically
            best_idx = np.argmax(evaluations)
            return unvisited[best_idx]
        else:
            # Exploration: Choose probabilistically based on the evaluation weights 
            probabilities = evaluations / np.sum(evaluations)
            return np.random.choice(unvisited, p=probabilities)

    def run(self, iterations, patience=None, known_optimum=None):
        """
        Runs the generic Ant-Q algorithm.
        - patience: Stops the algorithm if the best score hasn't improved for this many consecutive iterations.
        - known_optimum: Stops immediately if this score is reached.
        """
        best_overall_tour = None
        best_overall_length = float('inf')
        
        iterations_without_improvement = 0

        for it in range(iterations):
            tours = []
            unvisited_sets = []
            current_cities = []
            
            # Step 1: Initialization
            for k in range(self.m):
                start_city = random.randint(0, self.n - 1)
                current_cities.append(start_city)
                tours.append([start_city])
                unvisited = list(range(self.n))
                unvisited.remove(start_city)
                unvisited_sets.append(unvisited)
                
            # Step 2: Agents build their tours
            for step in range(self.n):
                for k in range(self.m):
                    r = current_cities[k]
                    if step < self.n - 1:
                        s = self._action_choice(r, unvisited_sets[k])
                        unvisited_sets[k].remove(s)
                    else:
                        s = tours[k][0]
                    
                    tours[k].append(s)
                    current_cities[k] = s
                    
                    if step < self.n - 1:
                        max_aq_next = max([self.AQ[s, z] for z in unvisited_sets[k]]) if unvisited_sets[k] else 0
                    else:
                        max_aq_next = 0
                        
                    self.AQ[r, s] = (1 - self.alpha) * self.AQ[r, s] + self.alpha * self.gamma * max_aq_next

            # Step 3: Compute delayed reinforcement
            tour_lengths = []
            for tour in tours:
                length = sum(self.distances[tour[i], tour[i+1]] for i in range(self.n))
                tour_lengths.append(length)
            
            best_idx = np.argmin(tour_lengths)
            iter_best_tour = tours[best_idx]
            iter_best_length = tour_lengths[best_idx]
            
            # --- Check for improvements and early stopping ---
            if iter_best_length < best_overall_length:
                best_overall_length = iter_best_length
                best_overall_tour = iter_best_tour
                iterations_without_improvement = 0 # Reset patience counter
            else:
                iterations_without_improvement += 1
                
            delta_AQ = self.W / iter_best_length
            for i in range(self.n):
                r, s = iter_best_tour[i], iter_best_tour[i+1]
                self.AQ[r, s] = (1 - self.alpha) * self.AQ[r, s] + self.alpha * delta_AQ
                
            # Print progress occasionally
            if (it + 1) % 10 == 0 or it == 0:
                print(f"Iteration {it + 1:3d} | Best Length: {best_overall_length} | Patience: {iterations_without_improvement}")
                
            # Condition 1: Reached known optimum
            if known_optimum is not None and best_overall_length <= known_optimum:
                print(f"\nStopping early: Known optimum ({known_optimum}) reached at iteration {it + 1}.")
                break
                
            # Condition 2: No improvement for 'patience' iterations
            if patience is not None and iterations_without_improvement >= patience:
                print(f"\nStopping early: No improvement for {patience} iterations (stopped at iteration {it + 1}).")
                break
                
        return best_overall_tour, best_overall_length
    


def calculate_euclidean_distance_matrix(coords):
    """
    Calculates a 2D Euclidean distance matrix from a list of (x, y) coordinates.
    Rounds to the nearest integer as is standard practice in TSPLIB EUC_2D problems.
    """
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Standard TSPLIB Euclidean distance calculation
                dist = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                matrix[i][j] = round(dist)
            else:
                matrix[i][j] = np.inf
    return matrix

def load_tsplib_euc2d(filepath):
    """
    Parses a standard TSPLIB file with NODE_COORD_SECTION (EUC_2D).
    Returns the distance matrix and the dimension of the problem.
    """
    coords = []
    reading_nodes = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "EOF":
                break
            if line == "NODE_COORD_SECTION":
                reading_nodes = True
                continue
            if reading_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    # parts[0] is the node ID, parts[1] is x, parts[2] is y
                    coords.append((float(parts[1]), float(parts[2])))
                    
    if not coords:
        raise ValueError("Could not find any coordinates. Ensure the file is a standard EUC_2D TSPLIB file.")
        
    return calculate_euclidean_distance_matrix(coords)

def main():
    # 1. Load the TSP Instance
    tsplib_file_path = "tsplib/st70.tsp"
    
    try:
        print(f"Attempting to load TSPLIB file: {tsplib_file_path}")
        distance_matrix = load_tsplib_euc2d(tsplib_file_path)
        print(f"Successfully loaded {len(distance_matrix)} cities.")
    except FileNotFoundError:
        print(f"File '{tsplib_file_path}' not found. Falling back to a 10-city mock problem.")
        # Fallback: Generate a random 10-city problem
        np.random.seed(42) # For reproducibility
        mock_coords = np.random.rand(10, 2) * 100 
        distance_matrix = calculate_euclidean_distance_matrix(mock_coords)

    # 2. Initialize the Ant-Q Algorithm
    # Parameters are set based on the experimental defaults in the 1995 paper
    n_cities = len(distance_matrix)
    
    ant_q = AntQ(
        distance_matrix=distance_matrix,
        m=n_cities,   # m=n: one agent per city
        alpha=0.1,    # Learning step
        gamma=0.3,    # Discount factor
        delta=1,      # Learned value weight
        beta=2,       # Heuristic value weight
        q0=0.9,       # Exploitation vs Exploration factor
        W=10          # Delayed reinforcement weight
    )

    # 3. Run the Algorithm
    # Note: The number of iterations can be adjusted based on the size of the problem and desired runtime. 
    # For larger problems, more iterations may be needed to find good solutions, but this will increase execution time
    iterations = 10 * n_cities
    patience = 2 * n_cities

    # --- START TIMING ---
    start_time = time.perf_counter()
    
    print(f"\nRunning Ant-Q for {iterations} iterations...")
    best_tour, best_length = ant_q.run(iterations=iterations, patience=patience, known_optimum=None)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # 4. Output the Results
    print("\n" + "="*40)
    print("Optimization Complete")
    print("="*40)
    print(f"Best Tour Length Found: {best_length}")
    print(f"Best Tour Sequence:\n{best_tour}")
    print(f"Execution Time: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    main()