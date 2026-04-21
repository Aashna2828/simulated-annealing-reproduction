import numpy as np
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing
from tsp import TSP, generate_random_cities

def run_single_trial(n_cities, verbose=False):
    cities = generate_random_cities(n_cities)
    tsp = TSP(cities)
    
    random_tour = tsp.random_tour()
    random_cost = tsp.tour_cost(random_tour)
    
    greedy_tour = tsp.greedy_tour()
    greedy_cost = tsp.tour_cost(greedy_tour)
    
    
    
    sa = SimulatedAnnealing(
        initial_temp=n_cities * 1.0,
        cooling_rate=0.98,
        iterations_per_temp=n_cities*5,
        min_acceptance_rate=0.001,
        max_iterations=n_cities*5000
)
    
    best_tour, best_cost, history = sa.optimize(
        initial_solution=random_tour,
        cost_function=tsp.tour_cost,
        neighbor_function=tsp.generate_neighbor_2opt,
        verbose=verbose
    )
    
    return {
        'random': random_cost,
        'greedy': greedy_cost,
        'sa': best_cost,
        'tour': best_tour,
        'tsp': tsp,
        'history': history
    }

def run_experiment(n_cities_list=[10, 25, 50, 100, 200], n_runs=10):
    print(f"\n{'='*70}")
    print(f"IMPROVED TSP EXPERIMENT - SCALED PARAMETERS")
    print(f"Problem sizes: {n_cities_list}")
    print(f"Runs per size: {n_runs}")
    print(f"{'='*70}\n")
    
    results = {n: {'random': [], 'greedy': [], 'sa': []} for n in n_cities_list}
    
    for n_cities in n_cities_list:
        print(f"\n{'='*70}")
        print(f"PROBLEM SIZE: {n_cities} cities")
        print(f"T0={max(10.0, n_cities/10.0)}, iters/temp={max(100, n_cities*2)}")
        print(f"{'='*70}")
        
        for run in range(n_runs):
            print(f"Run {run+1}/{n_runs}...", end=' ')
            
            result = run_single_trial(n_cities, verbose=False)
            
            results[n_cities]['random'].append(result['random'])
            results[n_cities]['greedy'].append(result['greedy'])
            results[n_cities]['sa'].append(result['sa'])
            
            alpha = result['sa'] / np.sqrt(n_cities)
            
            print(f"Greedy: {result['greedy']:.2f}, SA: {result['sa']:.2f}, alpha: {alpha:.4f}")
        
        print(f"\n--- Summary for {n_cities} cities ---")
        print(f"Random: {np.mean(results[n_cities]['random']):.2f} +/- {np.std(results[n_cities]['random']):.2f}")
        print(f"Greedy: {np.mean(results[n_cities]['greedy']):.2f} +/- {np.std(results[n_cities]['greedy']):.2f}")
        print(f"SA:     {np.mean(results[n_cities]['sa']):.2f} +/- {np.std(results[n_cities]['sa']):.2f}")
        
        avg_sa = np.mean(results[n_cities]['sa'])
        avg_greedy = np.mean(results[n_cities]['greedy'])
        improvement = 100 * (avg_greedy - avg_sa) / avg_greedy
        alpha_avg = avg_sa / np.sqrt(n_cities)
        
        print(f"SA improvement over greedy: {improvement:.1f}%")
        print(f"SA step length alpha: {alpha_avg:.4f} (paper target: < 1.0)")
    
    create_summary_table(results, n_cities_list)
    plot_results(results, n_cities_list)
    
    return results

def create_summary_table(results, n_cities_list):
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Size':<8} {'Greedy':<12} {'SA':<12} {'Improve':<12} {'Alpha':<10}")
    print(f"{'-'*70}")
    
    for n in n_cities_list:
        avg_greedy = np.mean(results[n]['greedy'])
        avg_sa = np.mean(results[n]['sa'])
        improvement = 100 * (avg_greedy - avg_sa) / avg_greedy
        alpha = avg_sa / np.sqrt(n)
        
        print(f"{n:<8} {avg_greedy:<12.2f} {avg_sa:<12.2f} {improvement:<12.1f}% {alpha:<10.4f}")
    
    print(f"{'-'*70}")
    print("Paper reports: alpha ~0.95 for SA, ~1.12 for greedy")
    print(f"{'='*70}\n")

def plot_results(results, n_cities_list):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for i, n in enumerate(n_cities_list):
        positions = [i*4 + 1, i*4 + 2, i*4 + 3]
        data = [results[n]['random'], results[n]['greedy'], results[n]['sa']]
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                       tick_labels=['Rand', 'Greedy', 'SA'] if i == 0 else ['', '', ''])
        
        colors = ['lightcoral', 'lightyellow', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax.set_xlabel('Problem Size')
    ax.set_ylabel('Tour Cost')
    ax.set_title('Comparison of Methods Across Problem Sizes')
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.set_xticklabels(n_cities_list)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    alphas_greedy = []
    alphas_sa = []
    
    for n in n_cities_list:
        alpha_g = np.mean(results[n]['greedy']) / np.sqrt(n)
        alpha_s = np.mean(results[n]['sa']) / np.sqrt(n)
        alphas_greedy.append(alpha_g)
        alphas_sa.append(alpha_s)
    
    ax.plot(n_cities_list, alphas_greedy, 'o-', label='Greedy', linewidth=2, markersize=8)
    ax.plot(n_cities_list, alphas_sa, 's-', label='SA', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Target', alpha=0.7)
    ax.axhline(y=0.95, color='green', linestyle='--', label='Paper', alpha=0.7)
    
    ax.set_xlabel('Number of Cities')
    ax.set_ylabel('Step Length alpha')
    ax.set_title('Step Length vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    improvements = []
    
    for n in n_cities_list:
        avg_greedy = np.mean(results[n]['greedy'])
        avg_sa = np.mean(results[n]['sa'])
        improvement = 100 * (avg_greedy - avg_sa) / avg_greedy
        improvements.append(improvement)
    
    ax.bar(range(len(n_cities_list)), improvements, color='steelblue', alpha=0.7)
    ax.set_xlabel('Problem Size')
    ax.set_ylabel('Improvement over Greedy (%)')
    ax.set_title('SA Improvement Over Greedy Heuristic')
    ax.set_xticks(range(len(n_cities_list)))
    ax.set_xticklabels(n_cities_list)
    ax.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    
    means_random = [np.mean(results[n]['random']) for n in n_cities_list]
    means_greedy = [np.mean(results[n]['greedy']) for n in n_cities_list]
    means_sa = [np.mean(results[n]['sa']) for n in n_cities_list]
    
    ax.plot(n_cities_list, means_random, 'o-', label='Random', linewidth=2, markersize=8)
    ax.plot(n_cities_list, means_greedy, 's-', label='Greedy', linewidth=2, markersize=8)
    ax.plot(n_cities_list, means_sa, '^-', label='SA', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Cities')
    ax.set_ylabel('Mean Tour Cost')
    ax.set_title('Solution Quality vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_improved_results.png', dpi=150, bbox_inches='tight')
    print("Saved plot: tsp_improved_results.png")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    print("\nStarting improved TSP experiment with scaled parameters")
    
    results = run_experiment(
        n_cities_list=[10, 25, 50, 100, 200],
        n_runs=10
    )
    
    print("\nExperiment complete")