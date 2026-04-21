import numpy as np
import matplotlib.pyplot as plt

class CircuitPartitioner:
    def __init__(self, n_circuits, connectivity_matrix, lambda_balance=0.01):
        self.n = n_circuits
        self.connectivity = connectivity_matrix
        self.lambda_balance = lambda_balance
    
    def cost(self, partition):
        mu = partition
        
        crossings = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.connectivity[i, j] > 0:
                    crossings += self.connectivity[i, j] * (mu[i] - mu[j])**2 / 2
        
        balance = np.sum(mu)**2
        
        total_cost = crossings + self.lambda_balance * balance
        
        return total_cost
    
    def count_pins(self, partition):
        mu = partition
        pins = 0
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.connectivity[i, j] > 0 and mu[i] != mu[j]:
                    pins += self.connectivity[i, j]
        
        return pins
    
    def flip_move(self, partition):
        new_partition = partition.copy()
        idx = np.random.randint(0, self.n)
        new_partition[idx] *= -1
        return new_partition
    
    def random_partition(self):
        return np.random.choice([-1, 1], size=self.n)
    
    def greedy_partition(self):
        partition = np.ones(self.n)
        partition[:self.n//2] = -1
        
        improved = True
        while improved:
            improved = False
            current_cost = self.cost(partition)
            
            for i in range(self.n):
                partition[i] *= -1
                new_cost = self.cost(partition)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    partition[i] *= -1
        
        return partition

def generate_random_circuit(n_circuits, avg_connections=4):
    connectivity = np.zeros((n_circuits, n_circuits))
    
    for i in range(n_circuits):
        n_connections = np.random.poisson(avg_connections)
        n_connections = min(n_connections, n_circuits - 1)
        
        targets = np.random.choice([j for j in range(n_circuits) if j != i], 
                                   size=n_connections, replace=False)
        
        for j in targets:
            weight = np.random.randint(1, 4)
            connectivity[i, j] = weight
            connectivity[j, i] = weight
    
    return connectivity

def run_partition_experiment(n_circuits=1000, n_runs=10):
    print(f"\n{'='*70}")
    print(f"CIRCUIT PARTITIONING EXPERIMENT")
    print(f"Circuits: {n_circuits}, Runs: {n_runs}")
    print(f"{'='*70}\n")
    
    random_pins = []
    greedy_pins = []
    sa_pins = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...")
        
        connectivity = generate_random_circuit(n_circuits, avg_connections=4)
        partitioner = CircuitPartitioner(n_circuits, connectivity, lambda_balance=0.01)
        
        random_part = partitioner.random_partition()
        random_p = partitioner.count_pins(random_part)
        random_pins.append(random_p)
        
        greedy_part = partitioner.greedy_partition()
        greedy_p = partitioner.count_pins(greedy_part)
        greedy_pins.append(greedy_p)
        
        from simulated_annealing import SimulatedAnnealing
        sa = SimulatedAnnealing(
            initial_temp=n_circuits / 50.0,
            cooling_rate=0.95,
            iterations_per_temp=n_circuits,
            min_acceptance_rate=0.001
        )
        
        best_part, best_cost, history = sa.optimize(
            initial_solution=random_part,
            cost_function=partitioner.cost,
            neighbor_function=partitioner.flip_move,
            verbose=False
        )
        
        sa_p = partitioner.count_pins(best_part)
        sa_pins.append(sa_p)
        
        print(f"  Random: {random_p} pins, Greedy: {greedy_p} pins, SA: {sa_p} pins")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"Random: {np.mean(random_pins):.0f} +/- {np.std(random_pins):.0f} pins")
    print(f"Greedy: {np.mean(greedy_pins):.0f} +/- {np.std(greedy_pins):.0f} pins")
    print(f"SA:     {np.mean(sa_pins):.0f} +/- {np.std(sa_pins):.0f} pins")
    
    improvement_over_greedy = 100 * (np.mean(greedy_pins) - np.mean(sa_pins)) / np.mean(greedy_pins)
    print(f"\nSA improvement over greedy: {improvement_over_greedy:.1f}%")
    print(f"Paper reports: ~53% improvement (700 pins -> 353 pins)")
    print(f"{'='*70}")
    
    plt.figure(figsize=(10, 6))
    data = [random_pins, greedy_pins, sa_pins]
    bp = plt.boxplot(data, tick_labels=['Random', 'Greedy', 'SA'], patch_artist=True)
    
    colors = ['lightcoral', 'lightyellow', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Number of Pins Required')
    plt.title(f'Circuit Partitioning ({n_circuits} circuits)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('circuit_partition_results.png', dpi=150)
    print("\nSaved plot: circuit_partition_results.png")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_partition_experiment(n_circuits=1000, n_runs=10)