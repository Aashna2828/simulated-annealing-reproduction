import numpy as np
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing
from tsp import TSP, generate_random_cities

def plot_convergence():
    print("Generating convergence plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('SA Convergence: Cost vs Temperature Steps', fontsize=14)
    
    sizes = [50, 100, 200]
    
    for idx, n in enumerate(sizes):
        print(f"Running N={n}...")
        
        cities = generate_random_cities(n)
        tsp = TSP(cities)
        
        greedy_cost = tsp.tour_cost(tsp.greedy_tour())
        
        sa = SimulatedAnnealing(
            initial_temp=n * 1.0,
            cooling_rate=0.98,
            iterations_per_temp=n*5,
            min_acceptance_rate=0.001,
            max_iterations=n*5000
        )
        
        _, best_cost, history = sa.optimize(
            tsp.random_tour(),
            tsp.tour_cost,
            tsp.generate_neighbor_2opt,
            verbose=False
        )
        
        ax = axes[idx]
        
        steps = range(len(history['best_cost_history']))
        
        ax.plot(steps, history['cost_history'], 
                alpha=0.4, color='blue', linewidth=0.8, label='Current cost')
        ax.plot(steps, history['best_cost_history'], 
                color='green', linewidth=2, label='Best cost')
        ax.axhline(y=greedy_cost, color='red', 
                   linestyle='--', linewidth=1.5, label=f'Greedy ({greedy_cost:.2f})')
        
        improvement = 100*(greedy_cost - best_cost)/greedy_cost
        ax.set_title(f'N={n} cities\nImprovement: {improvement:.1f}%')
        ax.set_xlabel('Temperature Steps')
        ax.set_ylabel('Tour Cost')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        print(f"  N={n}: Greedy={greedy_cost:.2f}, SA={best_cost:.2f}, "
              f"Improvement={improvement:.1f}%")
    
    plt.tight_layout()
    plt.savefig('convergence_plots.png', dpi=150, bbox_inches='tight')
    print("\nSaved: convergence_plots.png")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    plot_convergence()