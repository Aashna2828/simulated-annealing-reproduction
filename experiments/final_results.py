# import numpy as np
# import matplotlib.pyplot as plt
# from simulated_annealing import SimulatedAnnealing
# from tsp import TSP, generate_random_cities

# class CircuitPartitionFast:
#     def __init__(self, adjacency_matrix):
#         self.adj = adjacency_matrix
#         self.n_gates = len(adjacency_matrix)
#         self.lambda_balance = 0.01
    
#     def partition_cost(self, partition):
#     # vectorized - no nested loop
#         diff = partition[:, None] != partition[None, :]  # NxN boolean matrix
#         crossings = np.sum(self.adj * diff) / 2          # divide by 2 (symmetric)
#         imbalance = np.sum(partition)
#         cost = crossings + self.lambda_balance * imbalance * imbalance
#         return cost
    
#     def generate_neighbor(self, partition):
#         new_partition = partition.copy()
#         flip_idx = np.random.randint(0, self.n_gates)
#         new_partition[flip_idx] *= -1
#         return new_partition
    
#     def random_partition(self):
#         return np.random.choice([-1, 1], size=self.n_gates)

# def generate_random_circuit(n_gates, avg_connections):
#     adj = np.zeros((n_gates, n_gates))
#     for i in range(n_gates):
#         n_connections = np.random.poisson(avg_connections)
#         n_connections = min(n_connections, n_gates - 1)
#         if n_connections > 0:
#             targets = np.random.choice([j for j in range(n_gates) if j != i], 
#                                       size=n_connections, replace=False)
#             for j in targets:
#                 adj[i,j] = 1
#                 adj[j,i] = 1
#     return adj

# print("="*70)
# print("FAST FINAL RESULTS COLLECTION")
# print("="*70)

# np.random.seed(42)

# print("\n1. TSP RESULTS")
# print("-"*70)

# tsp_sizes = [50, 100, 200]
# tsp_results = []

# for n in tsp_sizes:
#     print(f"\nTSP {n} cities (3 runs):")
#     greedy_costs = []
#     sa_costs = []
    
#     for run in range(3):
#         cities = generate_random_cities(n)
#         tsp = TSP(cities)
        
#         greedy_tour = tsp.greedy_tour()
#         greedy_cost = tsp.tour_cost(greedy_tour)
#         greedy_costs.append(greedy_cost)
        
#         sa = SimulatedAnnealing(
#             initial_temp=n * 1.0,
#             cooling_rate=0.98,
#             iterations_per_temp=n*5,
#             min_acceptance_rate=0.001,
#             max_iterations=n*5000
# )
        
#         best, cost, history = sa.optimize(
#             tsp.random_tour(),
#             tsp.tour_cost,
#             tsp.generate_neighbor_2opt,
#             verbose=False
#         )
#         sa_costs.append(cost)
        
#         print(f"  Run {run+1}: Greedy={greedy_cost:.1f}, SA={cost:.1f}")
    
#     avg_g = np.mean(greedy_costs)
#     avg_s = np.mean(sa_costs)
#     improvement = 100*(avg_g - avg_s)/avg_g
#     alpha = avg_s / np.sqrt(n)
    
#     print(f"Average: Greedy={avg_g:.1f}, SA={avg_s:.1f}")
#     print(f"Improvement: {improvement:.1f}%, Alpha: {alpha:.3f}")
    
#     tsp_results.append({
#         'n': n,
#         'greedy': avg_g,
#         'sa': avg_s,
#         'improvement': improvement,
#         'alpha': alpha
#     })

# print("\n2. CIRCUIT PARTITIONING RESULTS")
# print("-"*70)

# circuit_sizes = [500, 1000, 2000]
# circuit_results = []

# for n in circuit_sizes:
#     print(f"\nCircuit {n} gates (3 runs):")
#     random_costs = []
#     sa_costs = []
    
#     for run in range(3):
#         circuit = generate_random_circuit(n, 3)
#         cp = CircuitPartitionFast(circuit)
        
#         random_partition = cp.random_partition()
#         random_cost = cp.partition_cost(random_partition)
#         random_costs.append(random_cost)
        
#         sa = SimulatedAnnealing(
#             initial_temp=n/5,
#             cooling_rate=0.95,
#             iterations_per_temp=n//10,
#             max_iterations=n*30
#         )
        
#         best, cost, history = sa.optimize(
#             random_partition,
#             cp.partition_cost,
#             cp.generate_neighbor,
#             verbose=False
#         )
#         sa_costs.append(cost)
        
#         print(f"  Run {run+1}: Random={random_cost:.0f}, SA={cost:.0f}")
    
#     avg_r = np.mean(random_costs)
#     avg_s = np.mean(sa_costs)
#     improvement = 100*(avg_r - avg_s)/avg_r
    
#     print(f"Average: Random={avg_r:.0f}, SA={avg_s:.0f}, Improvement={improvement:.1f}%")
    
#     circuit_results.append({
#         'n': n,
#         'random': avg_r,
#         'sa': avg_s,
#         'improvement': improvement
#     })

# print("\n" + "="*70)
# print("FINAL SUMMARY - ALL RESULTS")
# print("="*70)

# print("\nTSP Results:")
# print(f"{'Size':<10} {'Greedy':<10} {'SA':<10} {'Improve':<10} {'Alpha':<10}")
# print("-"*50)
# for r in tsp_results:
#     print(f"{r['n']:<10} {r['greedy']:<10.1f} {r['sa']:<10.1f} {r['improvement']:<10.1f}% {r['alpha']:<10.3f}")

# print("\nCircuit Partitioning Results:")
# print(f"{'Size':<10} {'Random':<10} {'SA':<10} {'Improve':<10}")
# print("-"*40)
# for r in circuit_results:
#     print(f"{r['n']:<10} {r['random']:<10.0f} {r['sa']:<10.0f} {r['improvement']:<10.1f}%")

# print("\nPaper Benchmarks:")
# print("  Circuit partitioning: 53% improvement (you achieved: 40-60% range)")
# print("  TSP alpha: ~0.95 (target: < 1.0)")

# print("\n" + "="*70)
# print("Results saved - ready for final report")
# print("="*70)

# with open('results_summary.txt', 'w') as f:
#     f.write("SIMULATED ANNEALING REPRODUCTION RESULTS\n")
#     f.write("="*70 + "\n\n")
#     f.write("TSP Results:\n")
#     f.write(f"{'Size':<10} {'Greedy':<10} {'SA':<10} {'Improve':<10} {'Alpha':<10}\n")
#     f.write("-"*50 + "\n")
#     for r in tsp_results:
#         f.write(f"{r['n']:<10} {r['greedy']:<10.1f} {r['sa']:<10.1f} {r['improvement']:<10.1f}% {r['alpha']:<10.3f}\n")
    
#     f.write("\nCircuit Partitioning Results:\n")
#     f.write(f"{'Size':<10} {'Random':<10} {'SA':<10} {'Improve':<10}\n")
#     f.write("-"*40 + "\n")
#     for r in circuit_results:
#         f.write(f"{r['n']:<10} {r['random']:<10.0f} {r['sa']:<10.0f} {r['improvement']:<10.1f}%\n")
    
#     f.write("\nPaper Benchmarks:\n")
#     f.write("  Circuit: 53% improvement\n")
#     f.write("  TSP alpha: ~0.95\n")

# print("\nSaved results to: results_summary.txt")

import numpy as np
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing
from tsp import TSP, generate_random_cities

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
        # single pass greedy descent from random start
        # approximates paper's "rapid cooling" (T=0) baseline
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

print("="*70)
print("FAST FINAL RESULTS COLLECTION")
print("="*70)

np.random.seed(42)

print("\n1. TSP RESULTS")
print("-"*70)

tsp_sizes = [50, 100, 200]
tsp_results = []

for n in tsp_sizes:
    print(f"\nTSP {n} cities (3 runs):")
    greedy_costs = []
    sa_costs = []
    
    for run in range(3):
        cities = generate_random_cities(n)
        tsp = TSP(cities)
        
        greedy_tour = tsp.greedy_tour()
        greedy_cost = tsp.tour_cost(greedy_tour)
        greedy_costs.append(greedy_cost)
        
        sa = SimulatedAnnealing(
            initial_temp=n * 1.0,
            cooling_rate=0.98,
            iterations_per_temp=n*5,
            min_acceptance_rate=0.001,
            max_iterations=n*5000
        )
        
        best, cost, history = sa.optimize(
            tsp.random_tour(),
            tsp.tour_cost,
            tsp.generate_neighbor_2opt,
            verbose=False
        )
        sa_costs.append(cost)
        
        print(f"  Run {run+1}: Greedy={greedy_cost:.1f}, SA={cost:.1f}")
    
    avg_g = np.mean(greedy_costs)
    avg_s = np.mean(sa_costs)
    improvement = 100*(avg_g - avg_s)/avg_g
    alpha = avg_s / np.sqrt(n)
    
    print(f"Average: Greedy={avg_g:.1f}, SA={avg_s:.1f}")
    print(f"Improvement: {improvement:.1f}%, Alpha: {alpha:.3f}")
    
    tsp_results.append({
        'n': n,
        'greedy': avg_g,
        'sa': avg_s,
        'improvement': improvement,
        'alpha': alpha
    })

print("\n2. CIRCUIT PARTITIONING RESULTS")
print("-"*70)

circuit_sizes = [500, 1000, 2000]
circuit_results = []

for n in circuit_sizes:
    print(f"\nCircuit {n} gates (3 runs):")
    random_costs = []
    rapid_costs = []
    sa_costs = []
    
    for run in range(3):
        print(f"  Run {run+1}/3 - generating circuit...", flush=True)
        circuit = generate_random_circuit(n, 3)
        cp = CircuitPartitionFast(circuit)
        
        # random baseline
        random_partition = cp.random_partition()
        random_cost = cp.partition_cost(random_partition)
        random_costs.append(random_cost)
        
        # rapid cooling baseline - matches paper's comparison
        print(f"  Run {run+1}/3 - running rapid cooling...", flush=True)
        rapid_part = cp.rapid_cooling()
        rapid_cost = cp.partition_cost(rapid_part)
        rapid_costs.append(rapid_cost)
        
        # simulated annealing
        print(f"  Run {run+1}/3 - running SA...", flush=True)
        sa = SimulatedAnnealing(
            initial_temp=n * 1.0,
            cooling_rate=0.95,
            iterations_per_temp=n//5,
            min_acceptance_rate=0.001,
            max_iterations=n*200
        )
        
        best, cost, history = sa.optimize(
            random_partition,
            cp.partition_cost,
            cp.generate_neighbor,
            verbose=False
        )
        sa_costs.append(cost)
        
        print(f"  Run {run+1}: Random={random_cost:.0f}, "
              f"Rapid Cooling={rapid_cost:.0f}, SA={cost:.0f}")
    
    avg_r = np.mean(random_costs)
    avg_rc = np.mean(rapid_costs)
    avg_s = np.mean(sa_costs)
    
    improvement_vs_random = 100*(avg_r - avg_s)/avg_r
    improvement_vs_rapid = 100*(avg_rc - avg_s)/avg_rc
    
    print(f"Average: Random={avg_r:.0f}, Rapid Cooling={avg_rc:.0f}, SA={avg_s:.0f}")
    print(f"Improvement vs Random:       {improvement_vs_random:.1f}%")
    print(f"Improvement vs Rapid Cooling: {improvement_vs_rapid:.1f}% (paper comparison)")
    
    circuit_results.append({
        'n': n,
        'random': avg_r,
        'rapid_cooling': avg_rc,
        'sa': avg_s,
        'improvement_vs_random': improvement_vs_random,
        'improvement_vs_rapid': improvement_vs_rapid
    })

print("\n" + "="*70)
print("FINAL SUMMARY - ALL RESULTS")
print("="*70)

print("\nTSP Results:")
print(f"{'Size':<10} {'Greedy':<10} {'SA':<10} {'Improve':<10} {'Alpha':<10}")
print("-"*50)
for r in tsp_results:
    print(f"{r['n']:<10} {r['greedy']:<10.1f} {r['sa']:<10.1f} "
          f"{r['improvement']:<10.1f}% {r['alpha']:<10.3f}")

print("\nCircuit Partitioning Results:")
print(f"{'Size':<10} {'Random':<10} {'Rapid Cool':<12} {'SA':<10} "
      f"{'vs Random':<12} {'vs Rapid':<12}")
print("-"*70)
for r in circuit_results:
    print(f"{r['n']:<10} {r['random']:<10.0f} {r['rapid_cooling']:<12.0f} "
          f"{r['sa']:<10.0f} {r['improvement_vs_random']:<12.1f}% "
          f"{r['improvement_vs_rapid']:<12.1f}%")

print("\nPaper benchmark: SA vs rapid cooling = 53%")
print("TSP paper benchmark: alpha ~0.95")

print("\n" + "="*70)
print("Results saved - ready for final report")
print("="*70)

with open('results_summary.txt', 'w') as f:
    f.write("SIMULATED ANNEALING REPRODUCTION RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write("TSP Results:\n")
    f.write(f"{'Size':<10} {'Greedy':<10} {'SA':<10} {'Improve':<10} {'Alpha':<10}\n")
    f.write("-"*50 + "\n")
    for r in tsp_results:
        f.write(f"{r['n']:<10} {r['greedy']:<10.1f} {r['sa']:<10.1f} "
                f"{r['improvement']:<10.1f}% {r['alpha']:<10.3f}\n")
    
    f.write("\nCircuit Partitioning Results:\n")
    f.write(f"{'Size':<10} {'Random':<10} {'Rapid Cool':<12} {'SA':<10} "
            f"{'vs Random':<12} {'vs Rapid':<12}\n")
    f.write("-"*70 + "\n")
    for r in circuit_results:
        f.write(f"{r['n']:<10} {r['random']:<10.0f} {r['rapid_cooling']:<12.0f} "
                f"{r['sa']:<10.0f} {r['improvement_vs_random']:<12.1f}% "
                f"{r['improvement_vs_rapid']:<12.1f}%\n")
    
    f.write("\nPaper benchmark: SA vs rapid cooling = 53%\n")
    f.write("TSP paper benchmark: alpha ~0.95\n")

print("\nSaved results to: results_summary.txt")