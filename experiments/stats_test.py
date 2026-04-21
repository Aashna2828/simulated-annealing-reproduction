import numpy as np
from scipy import stats
from simulated_annealing import SimulatedAnnealing
from tsp import TSP, generate_random_cities

def run_stats_test():
    n_cities_list = [50, 100, 200]
    runs_per_size = {50: 10, 100: 10, 200: 5}
    
    print("="*70)
    print("STATISTICAL SIGNIFICANCE TESTS - SA vs GREEDY")
    print("="*70)
    
    for n in n_cities_list:
        n_runs = runs_per_size[n]
        greedy_costs = []
        sa_costs = []
        
        print(f"\nN={n} cities, {n_runs} runs...")
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=' ', flush=True)
            
            cities = generate_random_cities(n)
            tsp = TSP(cities)
            
            greedy_cost = tsp.tour_cost(tsp.greedy_tour())
            greedy_costs.append(greedy_cost)
            
            sa = SimulatedAnnealing(
                initial_temp=n * 1.0,
                cooling_rate=0.98,
                iterations_per_temp=n*5,
                min_acceptance_rate=0.001,
                max_iterations=n*5000
            )
            _, sa_cost, _ = sa.optimize(
                tsp.random_tour(),
                tsp.tour_cost,
                tsp.generate_neighbor_2opt,
                verbose=False
            )
            sa_costs.append(sa_cost)
            print(f"Greedy={greedy_cost:.3f}, SA={sa_cost:.3f}")
        
        t_stat, p_value = stats.ttest_ind(greedy_costs, sa_costs)
        improvement = 100*(np.mean(greedy_costs)-np.mean(sa_costs))/np.mean(greedy_costs)
        
        print(f"\n  Greedy: {np.mean(greedy_costs):.3f} +/- {np.std(greedy_costs):.3f}")
        print(f"  SA:     {np.mean(sa_costs):.3f} +/- {np.std(sa_costs):.3f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print(f"  Result: SIGNIFICANT (p < 0.05) ")
        else:
            print(f"  Result: NOT significant (p >= 0.05) ✗")

if __name__ == "__main__":
    np.random.seed(42)
    run_stats_test()