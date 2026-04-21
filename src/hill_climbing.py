import numpy as np
from tsp import TSP, generate_random_cities

class HillClimbing:
    def __init__(self, max_iterations=100000):
        self.max_iterations = max_iterations

    def optimize(self, initial_solution, cost_function, neighbor_function):
        current_solution = initial_solution.copy()
        current_cost = cost_function(current_solution)

        for _ in range(self.max_iterations):
            neighbor = neighbor_function(current_solution)
            neighbor_cost = cost_function(neighbor)

            # Only accept if strictly better (no probabilistic acceptance)
            if neighbor_cost < current_cost:
                current_solution = neighbor
                current_cost = neighbor_cost

        return current_solution, current_cost


def run_comparison(n_cities_list=[50, 100, 200], n_runs=10):
    print("SA vs Hill Climbing - TSP Comparison\n")
    print(f"{'Size':<8} {'Greedy':<10} {'HC':<10} {'SA':<10} {'HC Improve':<14} {'SA Improve':<14}")
    print("-" * 66)

    for n in n_cities_list:
        greedy_costs, hc_costs, sa_costs = [], [], []

        for _ in range(n_runs):
            cities = generate_random_cities(n)
            tsp = TSP(cities)

            greedy_tour = tsp.greedy_tour()
            greedy_cost = tsp.tour_cost(greedy_tour)
            greedy_costs.append(greedy_cost)

            # Hill Climbing
            hc = HillClimbing(max_iterations=n * 500)
            _, hc_cost = hc.optimize(
                tsp.random_tour(),
                tsp.tour_cost,
                tsp.generate_neighbor_2opt
            )
            hc_costs.append(hc_cost)

            # Simulated Annealing
            from simulated_annealing import SimulatedAnnealing
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

        avg_g  = np.mean(greedy_costs)
        avg_hc = np.mean(hc_costs)
        avg_sa = np.mean(sa_costs)
        hc_imp = 100 * (avg_g - avg_hc) / avg_g
        sa_imp = 100 * (avg_g - avg_sa) / avg_g

        print(f"{n:<8} {avg_g:<10.2f} {avg_hc:<10.2f} {avg_sa:<10.2f} "
              f"{hc_imp:<14.1f} {sa_imp:<14.1f}")

    print("\nPositive = better than greedy. SA should consistently beat HC.")


if __name__ == "__main__":
    np.random.seed(42)
    run_comparison()