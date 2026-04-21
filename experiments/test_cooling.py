import numpy as np
from simulated_annealing import SimulatedAnnealing
from tsp import TSP, generate_random_cities

def run_cooling_comparison(n_cities=100, n_runs=5):
    print("="*70)
    print(f"COOLING SCHEDULE COMPARISON - {n_cities} cities, {n_runs} runs")
    print("="*70)
    
    cooling_rates = [0.85, 0.90, 0.95, 0.98, 0.99]
    
    print(f"\n{'Rate':<8} {'Avg Cost':<12} {'Std':<10} {'Avg Iters':<12} {'Improvement':<12}")
    print("-"*60)
    
    cities_list = [generate_random_cities(n_cities) for _ in range(n_runs)]
    greedy_costs = []
    for cities in cities_list:
        tsp = TSP(cities)
        greedy_costs.append(tsp.tour_cost(tsp.greedy_tour()))
    avg_greedy = np.mean(greedy_costs)
    
    for rate in cooling_rates:
        sa_costs = []
        sa_iters = []
        
        for i, cities in enumerate(cities_list):
            tsp = TSP(cities)
            sa = SimulatedAnnealing(
                initial_temp=n_cities * 1.0,
                cooling_rate=rate,
                iterations_per_temp=n_cities*5,
                min_acceptance_rate=0.001,
                max_iterations=n_cities*5000
            )
            _, sa_cost, history = sa.optimize(
                tsp.random_tour(),
                tsp.tour_cost,
                tsp.generate_neighbor_2opt,
                verbose=False
            )
            sa_costs.append(sa_cost)
            sa_iters.append(history['total_iterations'])
        
        avg_cost = np.mean(sa_costs)
        std_cost = np.std(sa_costs)
        avg_iters = np.mean(sa_iters)
        improvement = 100*(avg_greedy - avg_cost)/avg_greedy
        
        print(f"{rate:<8} {avg_cost:<12.3f} {std_cost:<10.3f} {avg_iters:<12.0f} {improvement:<12.1f}%")
    
    print("-"*60)
    print(f"Greedy baseline: {avg_greedy:.3f}")
    print("\nKey insight: Slower cooling = better solution but more iterations")
    print("Rate 0.98 balances quality and speed (our chosen parameter)")

if __name__ == "__main__":
    np.random.seed(42)
    run_cooling_comparison()