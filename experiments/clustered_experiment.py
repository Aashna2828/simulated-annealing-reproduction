import numpy as np
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing
from tsp import TSP, generate_clustered_cities

def run_clustered_experiment():
    print("="*70)
    print("CLUSTERED CITIES EXPERIMENT")
    print("Reproducing Figure 9 from Kirkpatrick et al. (1983)")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate clustered cities - 9 regions like the paper
    cities = generate_clustered_cities(
        n_clusters=9,
        cities_per_cluster=22,  # ~200 cities total
        cluster_spread=0.08
    )
    
    n_cities = len(cities)
    print(f"\nGenerated {n_cities} cities in 9 clusters")
    
    tsp = TSP(cities)
    
    # Greedy baseline
    greedy_tour = tsp.greedy_tour()
    greedy_cost = tsp.tour_cost(greedy_tour)
    print(f"Greedy tour cost: {greedy_cost:.3f}")
    
    # Run SA
    print("Running SA...")
    sa = SimulatedAnnealing(
        initial_temp=n_cities * 1.0,
        cooling_rate=0.98,
        iterations_per_temp=n_cities*5,
        min_acceptance_rate=0.001,
        max_iterations=n_cities*5000
    )
    
    best_tour, best_cost, history = sa.optimize(
        tsp.random_tour(),
        tsp.tour_cost,
        tsp.generate_neighbor_2opt,
        verbose=False
    )
    
    improvement = 100*(greedy_cost - best_cost)/greedy_cost
    alpha = best_cost / np.sqrt(n_cities)
    
    print(f"SA best cost: {best_cost:.3f}")
    print(f"Improvement over greedy: {improvement:.1f}%")
    print(f"Step length alpha: {alpha:.3f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Clustered Cities TSP - Reproducing Kirkpatrick et al. Figure 9', 
                 fontsize=13)
    
    # Greedy tour
    ax = axes[0]
    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=20, zorder=3)
    for i in range(len(greedy_tour)):
        a = greedy_tour[i]
        b = greedy_tour[(i+1) % len(greedy_tour)]
        ax.plot([cities[a,0], cities[b,0]], 
                [cities[a,1], cities[b,1]], 
                'b-', linewidth=0.5, alpha=0.7)
    ax.set_title(f'Greedy Tour\nCost: {greedy_cost:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # SA tour
    ax = axes[1]
    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=20, zorder=3)
    for i in range(len(best_tour)):
        a = best_tour[i]
        b = best_tour[(i+1) % len(best_tour)]
        ax.plot([cities[a,0], cities[b,0]], 
                [cities[a,1], cities[b,1]], 
                'g-', linewidth=0.5, alpha=0.7)
    ax.set_title(f'SA Tour\nCost: {best_cost:.3f} ({improvement:.1f}% better)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustered_cities.png', dpi=150, bbox_inches='tight')
    print("\nSaved: clustered_cities.png")
    plt.show()

if __name__ == "__main__":
    run_clustered_experiment()