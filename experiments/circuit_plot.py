import numpy as np
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing

class CircuitPartitionFast:
    def __init__(self, adjacency_matrix):
        self.adj = adjacency_matrix
        self.n_gates = len(adjacency_matrix)
        self.lambda_balance = 0.01
    
    def partition_cost(self, partition):
        diff = partition[:, None] != partition[None, :]
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
        return np.random.choice([-1, 1], size=self.n_gates).astype(float)
    
    def rapid_cooling(self):
        partition = self.random_partition()
        current_cost = self.partition_cost(partition)
        for i in range(self.n_gates):
            partition[i] *= -1
            new_cost = self.partition_cost(partition)
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                partition[i] *= -1
        return partition

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

def run_circuit_plot():
    print("Generating circuit partitioning plots...")
    np.random.seed(42)
    
    sizes = [500, 1000, 2000]
    random_means = []
    rapid_means = []
    sa_means = []
    improvements_random = []
    improvements_rapid = []
    
    for n in sizes:
        print(f"Running {n} gates...", flush=True)
        random_costs = []
        rapid_costs = []
        sa_costs = []
        
        for run in range(3):
            print(f"  Run {run+1}/3...", flush=True)
            circuit = generate_random_circuit(n, 3)
            cp = CircuitPartitionFast(circuit)
            
            random_part = cp.random_partition()
            random_cost = cp.partition_cost(random_part)
            random_costs.append(random_cost)
            
            rapid_part = cp.rapid_cooling()
            rapid_cost = cp.partition_cost(rapid_part)
            rapid_costs.append(rapid_cost)
            
            sa = SimulatedAnnealing(
                initial_temp=n * 1.0,
                cooling_rate=0.95,
                iterations_per_temp=n//5,
                min_acceptance_rate=0.001,
                max_iterations=n*200
            )
            _, sa_cost, _ = sa.optimize(
                random_part,
                cp.partition_cost,
                cp.generate_neighbor,
                verbose=False
            )
            sa_costs.append(sa_cost)
            print(f"  Run {run+1}: Random={random_cost:.0f}, "
                  f"Rapid={rapid_cost:.0f}, SA={sa_cost:.0f}")
        
        avg_r = np.mean(random_costs)
        avg_rc = np.mean(rapid_costs)
        avg_s = np.mean(sa_costs)
        
        random_means.append(avg_r)
        rapid_means.append(avg_rc)
        sa_means.append(avg_s)
        
        improvements_random.append(100*(avg_r - avg_s)/avg_r)
        improvements_rapid.append(100*(avg_rc - avg_s)/avg_rc)
        
        print(f"  Improvement vs Random: {improvements_random[-1]:.1f}%")
        print(f"  Improvement vs Rapid:  {improvements_rapid[-1]:.1f}%")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Circuit Partitioning Results - SA vs Baselines', fontsize=14)
    
    # Plot 1: Bar chart comparing all three methods
    ax = axes[0]
    x = np.arange(len(sizes))
    width = 0.25
    
    bars1 = ax.bar(x - width, random_means, width, 
                   label='Random', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x, rapid_means, width, 
                   label='Rapid Cooling', color='lightyellow', 
                   alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, sa_means, width, 
                   label='SA', color='lightgreen', alpha=0.8)
    
    ax.set_xlabel('Circuit Size (gates)')
    ax.set_ylabel('Number of Crossings')
    ax.set_title('Crossings by Method and Circuit Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Improvement percentages
    ax = axes[1]
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, improvements_random, width,
                   label='vs Random', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, improvements_rapid, width,
                   label='vs Rapid Cooling (paper comparison)',
                   color='darkorange', alpha=0.8)
    
    ax.axhline(y=53, color='red', linestyle='--', 
               linewidth=2, label='Paper benchmark (53%)')
    
    ax.set_xlabel('Circuit Size (gates)')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('SA Improvement Over Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('circuit_partition_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: circuit_partition_results.png")
    plt.show()

if __name__ == "__main__":
    run_circuit_plot()