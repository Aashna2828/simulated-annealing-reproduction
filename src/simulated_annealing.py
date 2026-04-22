"""
Core Simulated Annealing Framework
Based on Kirkpatrick et al. (1983)
"""

import numpy as np
import math

class SimulatedAnnealing:
    """
    Generic Simulated Annealing optimizer.
    
    Uses Metropolis acceptance criterion:
    - Accept if cost decreases (ΔE ≤ 0)
    - Accept with probability exp(-ΔE/T) if cost increases
    """
    
    def __init__(self, 
                 initial_temp=10.0,
                 cooling_rate=0.9,
                 iterations_per_temp=100,
                 min_acceptance_rate=0.01,
                 max_iterations=100000):
        """
        Initialize SA parameters.
        
        Args:
            initial_temp: Starting temperature T_0
            cooling_rate: Cooling ratio (T_new = cooling_rate * T_old)
            iterations_per_temp: Moves to attempt at each temperature
            min_acceptance_rate: Stop if acceptance rate below this
            max_iterations: Maximum total iterations
        """
        self.T0 = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.min_acceptance_rate = min_acceptance_rate
        self.max_iterations = max_iterations
        
        # Track history
        self.cost_history = []
        self.temp_history = []
        self.best_cost_history = []
        
    def optimize(self, initial_solution, cost_function, neighbor_function, verbose=True):
        """
        Run simulated annealing optimization.
        
        Args:
            initial_solution: Starting configuration
            cost_function: Function that takes solution, returns cost
            neighbor_function: Function that takes solution, returns neighbor
            verbose: Print progress
            
        Returns:
            best_solution, best_cost, history
        """
        # Initialize
        current_solution = initial_solution
        current_cost = cost_function(current_solution)
        
        best_solution = current_solution.copy() if hasattr(current_solution, 'copy') else current_solution
        best_cost = current_cost
        
        temperature = self.T0
        total_iterations = 0
        temps_with_low_acceptance = 0
        
        if verbose:
            print(f"Starting SA: Initial cost = {current_cost:.2f}, T0 = {temperature:.2f}")
        
        
        while total_iterations < self.max_iterations:
            accepted_moves = 0
            
            # Iterations at this temperature
            for i in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor_solution = neighbor_function(current_solution)
                neighbor_cost = cost_function(neighbor_solution)
                
                # Calculate cost change
                delta_cost = neighbor_cost - current_cost
                
                
                if delta_cost <= 0:
                    
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
                    accepted_moves += 1
                    
                    if current_cost < best_cost:
                        best_solution = neighbor_solution.copy() if hasattr(neighbor_solution, 'copy') else neighbor_solution
                        best_cost = current_cost
                else:
                    
                    alpha = np.random.random()
                    if alpha <= math.exp(-delta_cost / temperature):
                        current_solution = neighbor_solution
                        current_cost = neighbor_cost
                        accepted_moves += 1
                        
                        if current_cost < best_cost:
                            best_solution = neighbor_solution.copy() if hasattr(neighbor_solution, 'copy') else neighbor_solution
                            best_cost = current_cost
                
                total_iterations += 1
            
            
            acceptance_rate = accepted_moves / self.iterations_per_temp
            
            # Track history
            self.cost_history.append(current_cost)
            self.temp_history.append(temperature)
            self.best_cost_history.append(best_cost)
            
            if verbose and total_iterations % 1000 == 0:
                print(f"Iter {total_iterations}: T={temperature:.4f}, Cost={current_cost:.2f}, "
                      f"Best={best_cost:.2f}, Accept={acceptance_rate:.1%}")
            
            # Check stopping criteria
            if acceptance_rate < self.min_acceptance_rate:
                temps_with_low_acceptance += 1
                if temps_with_low_acceptance >= 3:
                    if verbose:
                        print(f"Stopped: Low acceptance rate for 3 consecutive temperatures")
                    break
            else:
                temps_with_low_acceptance = 0
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Stop if temperature too low
            if temperature < 1e-8:
                if verbose:
                    print(f"Stopped: Temperature too low")
                break
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best cost: {best_cost:.2f}")
            print(f"Total iterations: {total_iterations}")
        
        return best_solution, best_cost, {
            'cost_history': self.cost_history,
            'temp_history': self.temp_history,
            'best_cost_history': self.best_cost_history,
            'total_iterations': total_iterations
        }