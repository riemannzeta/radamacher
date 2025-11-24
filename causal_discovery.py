import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Set

class CausalDiscovery:
    """
    Implements the PC (Peter-Clark) Algorithm for Causal Discovery
    to infer the causal graph structure from observational data.
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        self.data = data
        self.alpha = alpha
        self.variables = list(data.columns)
        self.n_samples = len(data)
        self.graph = nx.Graph()  # Start with undirected graph
        self.graph.add_nodes_from(self.variables)
        self.separating_sets = {}
        
    def _partial_correlation(self, x: str, y: str, z: List[str]) -> float:
        """
        Calculate partial correlation of x and y given z
        """
        if not z:
            return self.data[x].corr(self.data[y])
            
        # Linear regression residuals
        # This assumes linearity, which is a limitation but standard for basic PC
        # For more robust tests, we'd use non-parametric CI tests
        
        # Combine x, y, z
        all_vars = [x, y] + z
        subset = self.data[all_vars].dropna()
        
        if len(subset) < len(z) + 5:
            return 0.0  # Not enough data
            
        # Inverse covariance matrix method
        try:
            cov = subset.cov()
            inv_cov = np.linalg.pinv(cov)
            
            # Indices
            idx_x = 0
            idx_y = 1
            
            # Partial correlation formula from precision matrix
            p_corr = -inv_cov[idx_x, idx_y] / np.sqrt(inv_cov[idx_x, idx_x] * inv_cov[idx_y, idx_y])
            return p_corr
        except:
            return 0.0

    def _test_conditional_independence(self, x: str, y: str, z: List[str]) -> bool:
        """
        Test if x is independent of y given z using Fisher's z-transform
        Returns True if independent (p-value > alpha)
        """
        r = self._partial_correlation(x, y, z)
        
        # Fisher's z-transform
        if abs(r) >= 1.0: r = 0.99999 * np.sign(r)
        z_stat = 0.5 * np.log((1 + r) / (1 - r))
        
        # Standard error
        se = 1.0 / np.sqrt(self.n_samples - len(z) - 3)
        
        # Z-score
        statistic = np.abs(z_stat) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(statistic))
        
        return p_value > self.alpha

    def run_pc_algorithm(self):
        """
        Run the PC algorithm to learn the causal structure
        """
        print("Running PC Algorithm for Causal Discovery...")
        
        # Step 1: Start with complete undirected graph
        self.graph = nx.complete_graph(self.variables)
        
        # Step 2: Skeleton Discovery (Remove edges based on independence)
        depth = 0
        while True:
            adjacencies = {node: set(self.graph.neighbors(node)) for node in self.graph.nodes()}
            edges_to_remove = []
            
            # Check all edges
            for x, y in self.graph.edges():
                # Find neighbors of x (excluding y)
                neighbors = list(adjacencies[x] - {y})
                
                if len(neighbors) < depth:
                    continue
                
                # Check all subsets of size 'depth'
                for z in combinations(neighbors, depth):
                    z_list = list(z)
                    if self._test_conditional_independence(x, y, z_list):
                        print(f"  Edge {x}-{y} removed (indep given {z_list})")
                        edges_to_remove.append((x, y))
                        self.separating_sets[(x, y)] = z_list
                        self.separating_sets[(y, x)] = z_list
                        break
            
            # Remove edges
            for x, y in edges_to_remove:
                if self.graph.has_edge(x, y):
                    self.graph.remove_edge(x, y)
            
            # Stop if no nodes have enough neighbors for next depth
            max_degree = max([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0
            if max_degree <= depth:
                break
                
            depth += 1
            
        # Step 3: Orientation (V-structures)
        # X - Z - Y where X and Y are not connected
        # If Z is NOT in separating set of (X, Y), then X -> Z <- Y
        self.di_graph = nx.DiGraph()
        self.di_graph.add_nodes_from(self.variables)
        
        # Initialize with undirected edges
        for u, v in self.graph.edges():
            self.di_graph.add_edge(u, v)
            self.di_graph.add_edge(v, u) # Bidirectional initially
            
        # Find unshielded colliders
        for z in self.graph.nodes():
            neighbors = list(self.graph.neighbors(z))
            for x, y in combinations(neighbors, 2):
                if not self.graph.has_edge(x, y):
                    # Check if Z is in separating set
                    sep_set = self.separating_sets.get((x, y), [])
                    
                    if z not in sep_set:
                        print(f"  Found V-structure: {x} -> {z} <- {y}")
                        # Orient edges
                        if self.di_graph.has_edge(z, x): self.di_graph.remove_edge(z, x)
                        if self.di_graph.has_edge(z, y): self.di_graph.remove_edge(z, y)
                        
        # Step 4: Orientation Propagation (Meek Rules) - Simplified
        # Rule 1: X -> Y - Z => X -> Y -> Z (avoid new collider)
        changed = True
        while changed:
            changed = False
            for y in self.di_graph.nodes():
                parents = [p for p in self.di_graph.predecessors(y) if not self.di_graph.has_edge(y, p)]
                neighbors = [n for n in self.di_graph.successors(y) if self.di_graph.has_edge(n, y)] # Undirected
                
                for x in parents:
                    for z in neighbors:
                        if not self.di_graph.has_edge(x, z) and not self.di_graph.has_edge(z, x):
                            print(f"  Propagating orientation: {y} -> {z}")
                            self.di_graph.remove_edge(z, y)
                            changed = True
                            
        return self.di_graph

    def visualize_graph(self, title="Learned Causal Graph"):
        """Visualize the learned graph"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.di_graph, k=2)
        
        nx.draw(self.di_graph, pos, with_labels=True, 
                node_color='lightblue', node_size=2000, 
                arrowsize=20, font_size=10)
        
        plt.title(title)
        plt.savefig('learned_causal_graph.png')
        plt.close()

if __name__ == "__main__":
    # Test with dummy data
    df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
    cd = CausalDiscovery(df)
    graph = cd.run_pc_algorithm()
