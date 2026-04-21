"""
Traveling Salesman Problem implementation for Simulated Annealing.
"""

import numpy as np
import matplotlib.pyplot as plt

class TSP:
    """Traveling Salesman Problem representation and operations."""
    
    def __init__(self, cities):
        """
        Initialize TSP instance.
        
        Args:
            cities: numpy array of shape (n_cities, 2) with (x, y) coordinates
        """
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        
        # Precompute distance matrix for efficiency
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        """Compute pairwise distances between all cities."""
        n = self.n_cities
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Euclidean distance
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def tour_cost(self, tour):
        """
        Calculate total cost (length) of a tour.
        
        Args:
            tour: List or array of city indices (permutation of 0 to n-1)
            
        Returns:
            Total tour length
        """
        cost = 0.0
        n = len(tour)
        
        for i in range(n):
            city_a = tour[i]
            city_b = tour[(i + 1) % n]  # Wrap around to start
            cost += self.distance_matrix[city_a, city_b]
        
        return cost
    
    def generate_neighbor_2opt(self, tour):
        """
        
        Args:
            tour: Current tour
            
        Returns:
            New tour (neighbor)
        """
        tour = list(tour)
        n = len(tour)
        
        
        i, j = sorted(np.random.choice(n, size=2, replace=False))
        
        
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        
        return np.array(new_tour)
    
    def random_tour(self):
        """Generate a random tour (random permutation of cities)."""
        return np.random.permutation(self.n_cities)
    
    def greedy_tour(self):
        """
        Generate tour using greedy nearest-neighbor heuristic.
        (For comparison with SA)
        """
        unvisited = set(range(self.n_cities))
        tour = [0]  # Start at city 0
        unvisited.remove(0)
        
        current = 0
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return np.array(tour)
    
    def plot_tour(self, tour, title="TSP Tour", save_path=None):
        """Visualize a tour."""
        plt.figure(figsize=(8, 8))
        
        # Plot cities
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100, zorder=3)
        
        # Plot tour edges
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            plt.plot([self.cities[city_a, 0], self.cities[city_b, 0]],
                    [self.cities[city_a, 1], self.cities[city_b, 1]],
                    'b-', linewidth=1, zorder=1)
        
        # Mark start city
        start = tour[0]
        plt.scatter(self.cities[start, 0], self.cities[start, 1], 
                   c='green', s=200, marker='*', zorder=4, label='Start')
        
        plt.title(f"{title}\nCost: {self.tour_cost(tour):.2f}")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def generate_random_cities(n_cities, grid_size=None):
    """
    Generate random cities in a square.
    
    Paper uses square with side length N^(1/2) for normalization.
    
    Args:
        n_cities: Number of cities
        grid_size: Side length of square (default: sqrt(n_cities))
        
    Returns:
        numpy array of shape (n_cities, 2)
    """
    if grid_size is None:
        grid_size = 1.0
    
    cities = np.random.uniform(0, grid_size, size=(n_cities, 2))
    return cities


def generate_clustered_cities(n_clusters=9, cities_per_cluster=44, cluster_spread=0.1):
    """
    Generate clustered cities like Figure 9 in paper.
    
    Creates 9 regions with gaps between them.
    
    Args:
        n_clusters: Number of clusters (paper uses 9)
        cities_per_cluster: Cities in each cluster (paper uses ~44)
        cluster_spread: How spread out cities are within clusters
        
    Returns:
        numpy array of cities
    """
    # Arrange cluster centers in 3x3 grid
    grid_size = 3
    cluster_centers = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (i + 0.5) / grid_size
            center_y = (j + 0.5) / grid_size
            cluster_centers.append([center_x, center_y])
    
    # Generate cities around each center
    cities = []
    for center in cluster_centers[:n_clusters]:
        for _ in range(cities_per_cluster):
            # Random offset from center
            offset = np.random.randn(2) * cluster_spread
            city = np.array(center) + offset
            cities.append(city)
    
    return np.array(cities)