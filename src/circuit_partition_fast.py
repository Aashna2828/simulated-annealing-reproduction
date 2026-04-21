import numpy as np

class CircuitPartitionFast:
    def __init__(self, adjacency_matrix):
        self.adj = adjacency_matrix
        self.n_gates = len(adjacency_matrix)
        self.lambda_balance = 0.01
    
    def partition_cost(self, partition):
    
        diff = partition[:, None] != partition[None, :]  # NxN boolean matrix
        crossings = np.sum(self.adj * diff) / 2          
        imbalance = np.sum(partition)
        cost = crossings + self.lambda_balance * imbalance * imbalance
        return cost
    
    def generate_neighbor(self, partition):
        new_partition = partition.copy()
        flip_idx = np.random.randint(0, self.n_gates)
        new_partition[flip_idx] *= -1
        return new_partition
    
    def random_partition(self):
        return np.random.choice([-1, 1], size=self.n_gates)
    
    def count_pins(self, partition):
        pins_a = 0
        pins_b = 0
        
        for i in range(self.n_gates):
            for j in range(self.n_gates):
                if self.adj[i,j] > 0 and partition[i] != partition[j]:
                    if partition[i] == -1:
                        pins_a += 1
                    else:
                        pins_b += 1
                    break
        
        return pins_a, pins_b

def generate_random_circuit(n_gates, avg_connections):
    adj = np.zeros((n_gates, n_gates))
    
    for i in range(n_gates):
        n_connections = np.random.poisson(avg_connections)
        n_connections = min(n_connections, n_gates - 1)
        
        if n_connections > 0:
            targets = np.random.choice([j for j in range(n_gates) if j != i], 
                                      size=n_connections, replace=False)
            for j in targets:
                adj[i,j] = 1
                adj[j,i] = 1
    
    return adj

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Quick test of circuit partitioning\n")
    
    circuit = generate_random_circuit(100, 3)
    cp = CircuitPartitionFast(circuit)
    
    random_partition = cp.random_partition()
    random_cost = cp.partition_cost(random_partition)
    
    print(f"Random partition cost: {random_cost:.0f}")
    
    from simulated_annealing import SimulatedAnnealing
    
    sa = SimulatedAnnealing(
        initial_temp=100.0,
        cooling_rate=0.95,
        iterations_per_temp=50,
        max_iterations=5000
    )
    
    best, cost, history = sa.optimize(
        random_partition,
        cp.partition_cost,
        cp.generate_neighbor,
        verbose=True
    )
    
    print(f"\nFinal cost: {cost:.0f}")
    print(f"Improvement: {100*(random_cost-cost)/random_cost:.1f}%")