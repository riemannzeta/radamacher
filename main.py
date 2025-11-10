import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class TemporalModel:
    """Base class for temporal causal models"""
    name: str
    max_lag: int
    n_parameters: int
    
class RademacherComplexityAnalyzer:
    """
    Analyzes Rademacher complexity of different temporal causal models
    to test whether tracking detailed relatedness improves generalization
    """
    
    def __init__(self, n_samples: int = 1000, n_rademacher: int = 100):
        self.n_samples = n_samples
        self.n_rademacher = n_rademacher
        
    def compute_empirical_rademacher(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    model_class,
                                    model_params: Dict) -> float:
        """
        Compute empirical Rademacher complexity for a given model class
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            model_class: Model class to evaluate
            model_params: Parameters for model initialization
        
        Returns:
            Empirical Rademacher complexity
        """
        n = len(y)
        rademacher_sum = 0.0
        
        for _ in range(self.n_rademacher):
            # Generate Rademacher variables (+1 or -1 with equal probability)
            sigma = np.random.choice([-1, 1], size=n)
            
            # Fit model to Rademacher-labeled data
            model = model_class(**model_params)
            model.fit(X, sigma)
            
            # Compute correlation with random labels
            predictions = model.predict(X)
            correlation = np.mean(sigma * predictions)
            rademacher_sum += np.abs(correlation)
        
        return rademacher_sum / self.n_rademacher

class KinSelectionModel:
    """
    Complex model tracking all pairwise relatedness with time lags
    Represents inclusive fitness approach with detailed kinship tracking
    """
    
    def __init__(self, max_lag: int = 10, n_individuals: int = 100, 
                 regularization: float = 0.01):
        self.max_lag = max_lag
        self.n_individuals = n_individuals
        self.regularization = regularization
        # Track all pairwise interactions across all lags
        self.n_parameters = n_individuals * (n_individuals - 1) * max_lag // 2
        self.model = None
        
    def create_relatedness_features(self, X: np.ndarray, 
                                   relatedness_matrix: np.ndarray) -> np.ndarray:
        """
        Create features tracking all pairwise relatedness at different lags
        
        Args:
            X: Raw time series data (n_timepoints, n_individuals)
            relatedness_matrix: Pairwise relatedness (n_individuals, n_individuals)
        
        Returns:
            Feature matrix with lagged relatedness interactions
        """
        n_time, n_ind = X.shape
        features = []
        
        for t in range(self.max_lag, n_time):
            row_features = []
            
            # For each pair of individuals
            for i in range(n_ind):
                for j in range(i+1, n_ind):
                    # For each lag
                    for lag in range(1, self.max_lag + 1):
                        if t - lag >= 0:
                            # Interaction weighted by relatedness
                            interaction = (X[t-lag, i] * X[t-lag, j] * 
                                         relatedness_matrix[i, j])
                            row_features.append(interaction)
            
            features.append(row_features)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the kin selection model"""
        # Simulate relatedness matrix (in reality, this would be measured)
        relatedness = self._generate_relatedness_matrix()
        
        # Create complex feature set
        features = self.create_relatedness_features(X, relatedness)
        
        # Ensure y is aligned with features
        y_aligned = y[self.max_lag:]
        
        # Use Ridge regression to handle high dimensionality
        self.model = Ridge(alpha=self.regularization)
        self.model.fit(features, y_aligned)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        relatedness = self._generate_relatedness_matrix()
        features = self.create_relatedness_features(X, relatedness)
        return self.model.predict(features)
    
    def _generate_relatedness_matrix(self) -> np.ndarray:
        """Generate a simulated relatedness matrix"""
        n = self.n_individuals
        # Create symmetric matrix with values between 0 and 1
        matrix = np.random.beta(2, 5, (n, n))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)  # Self-relatedness = 1
        return matrix

class GroupSelectionModel:
    """
    Simple model tracking colony-level properties
    Represents mean field approximation ignoring detailed relatedness
    """
    
    def __init__(self, max_lag: int = 10, n_groups: int = 10,
                 regularization: float = 0.01):
        self.max_lag = max_lag
        self.n_groups = n_groups
        self.regularization = regularization
        # Only track group-level summaries
        self.n_parameters = n_groups * max_lag * 3  # mean, var, size per group
        self.model = None
        
    def create_group_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create features tracking group-level statistics
        
        Args:
            X: Raw time series data (n_timepoints, n_individuals)
        
        Returns:
            Feature matrix with group-level statistics at different lags
        """
        n_time, n_ind = X.shape
        features = []
        
        # Assign individuals to groups (in reality, based on colony membership)
        group_size = n_ind // self.n_groups
        
        for t in range(self.max_lag, n_time):
            row_features = []
            
            for g in range(self.n_groups):
                start_idx = g * group_size
                end_idx = min((g + 1) * group_size, n_ind)
                
                for lag in range(1, self.max_lag + 1):
                    if t - lag >= 0:
                        group_data = X[t-lag, start_idx:end_idx]
                        # Track group-level statistics
                        row_features.extend([
                            np.mean(group_data),  # Group mean
                            np.var(group_data),   # Group variance  
                            len(group_data)       # Group size
                        ])
            
            features.append(row_features)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the group selection model"""
        features = self.create_group_features(X)
        y_aligned = y[self.max_lag:]
        
        self.model = Ridge(alpha=self.regularization)
        self.model.fit(features, y_aligned)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        features = self.create_group_features(X)
        return self.model.predict(features)

class TemporalCausalityExperiment:
    """
    Main experiment comparing complexity and generalization of different
    temporal causal models, testing whether detailed kinship tracking
    improves prediction beyond simpler group-level models
    """
    
    def __init__(self, n_individuals: int = 100, n_timepoints: int = 500,
                 n_groups: int = 10):
        self.n_individuals = n_individuals
        self.n_timepoints = n_timepoints
        self.n_groups = n_groups
        
    def generate_synthetic_data(self, 
                               true_model: str = "group",
                               noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic evolutionary dynamics data
        
        Args:
            true_model: "kin" for kin selection dynamics, "group" for group selection
            noise_level: Amount of noise to add
        
        Returns:
            X: Time series of individual states (n_timepoints, n_individuals)
            y: Target variable (e.g., colony fitness)
        """
        np.random.seed(42)
        
        # Initialize population
        X = np.random.randn(self.n_timepoints, self.n_individuals)
        
        if true_model == "group":
            # Group selection dynamics - fitness depends on group averages
            y = np.zeros(self.n_timepoints)
            group_size = self.n_individuals // self.n_groups
            
            for t in range(1, self.n_timepoints):
                group_effects = []
                for g in range(self.n_groups):
                    start = g * group_size
                    end = min((g + 1) * group_size, self.n_individuals)
                    group_mean = np.mean(X[t-1, start:end])
                    group_effects.append(group_mean)
                
                # Colony fitness is function of group cooperation
                y[t] = np.mean(group_effects) + np.random.randn() * noise_level
                
                # Update individual states based on group dynamics
                for i in range(self.n_individuals):
                    group_idx = i // group_size
                    X[t, i] = 0.8 * X[t-1, i] + 0.2 * group_effects[group_idx]
                    
        elif true_model == "kin":
            # Kin selection dynamics - fitness depends on relatedness
            relatedness = self._generate_relatedness_matrix()
            y = np.zeros(self.n_timepoints)
            
            for t in range(1, self.n_timepoints):
                # Hamilton's rule effects
                for i in range(self.n_individuals):
                    kin_effect = 0
                    for j in range(self.n_individuals):
                        if i != j:
                            kin_effect += relatedness[i, j] * X[t-1, j]
                    
                    X[t, i] = 0.8 * X[t-1, i] + 0.2 * kin_effect / self.n_individuals
                
                # Colony fitness depends on inclusive fitness
                y[t] = np.mean(X[t]) + np.random.randn() * noise_level
        
        else:
            raise ValueError(f"Unknown model type: {true_model}")
        
        return X, y
    
    def _generate_relatedness_matrix(self) -> np.ndarray:
        """Generate relatedness matrix for the population"""
        n = self.n_individuals
        matrix = np.random.beta(2, 5, (n, n))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        return matrix
    
    def run_complexity_comparison(self, 
                                 max_lags: List[int] = [5, 10, 20],
                                 n_runs: int = 10) -> pd.DataFrame:
        """
        Compare Rademacher complexity and generalization across models
        
        Args:
            max_lags: Different lag lengths to test
            n_runs: Number of experimental runs
        
        Returns:
            DataFrame with results
        """
        results = []
        
        for run in tqdm(range(n_runs), desc="Running experiments"):
            for true_model in ["group", "kin"]:
                # Generate data according to true model
                X, y = self.generate_synthetic_data(true_model=true_model)
                
                # Split into train and test
                split_point = int(0.7 * self.n_timepoints)
                X_train, X_test = X[:split_point], X[split_point:]
                y_train, y_test = y[:split_point], y[split_point:]
                
                for max_lag in max_lags:
                    # Test both model types
                    models = {
                        "kin_selection": KinSelectionModel(
                            max_lag=max_lag,
                            n_individuals=self.n_individuals
                        ),
                        "group_selection": GroupSelectionModel(
                            max_lag=max_lag,
                            n_groups=self.n_groups
                        )
                    }
                    
                    analyzer = RademacherComplexityAnalyzer()
                    
                    for model_name, model in models.items():
                        # Compute Rademacher complexity
                        if model_name == "kin_selection":
                            features = model.create_relatedness_features(
                                X_train, 
                                model._generate_relatedness_matrix()
                            )
                        else:
                            features = model.create_group_features(X_train)
                        
                        y_train_aligned = y_train[max_lag:]
                        
                        rademacher = analyzer.compute_empirical_rademacher(
                            features, y_train_aligned,
                            Ridge, {"alpha": 0.01}
                        )
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Test generalization
                        try:
                            y_pred = model.predict(X_test)
                            y_test_aligned = y_test[max_lag:]
                            
                            # Ensure arrays have same length
                            min_len = min(len(y_pred), len(y_test_aligned))
                            test_error = mean_squared_error(
                                y_test_aligned[:min_len], 
                                y_pred[:min_len]
                            )
                        except Exception as e:
                            print(f"Error in prediction: {e}")
                            test_error = np.nan
                        
                        results.append({
                            "run": run,
                            "true_model": true_model,
                            "test_model": model_name,
                            "max_lag": max_lag,
                            "n_parameters": model.n_parameters,
                            "rademacher_complexity": rademacher,
                            "test_error": test_error,
                            "complexity_per_param": rademacher / model.n_parameters
                        })
        
        return pd.DataFrame(results)
    
    def plot_results(self, results_df: pd.DataFrame):
        """
        Visualize the complexity-generalization tradeoff
        
        Args:
            results_df: DataFrame with experimental results
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Group by true model
        for idx, true_model in enumerate(["group", "kin"]):
            data = results_df[results_df["true_model"] == true_model]
            
            # Plot 1: Rademacher complexity vs parameters
            ax = axes[idx, 0]
            for model_type in data["test_model"].unique():
                model_data = data[data["test_model"] == model_type]
                grouped = model_data.groupby("max_lag").mean(numeric_only=True)
                ax.plot(grouped["n_parameters"],
                       grouped["rademacher_complexity"],
                       marker='o', label=model_type)
            ax.set_xlabel("Number of Parameters")
            ax.set_ylabel("Rademacher Complexity")
            ax.set_title(f"Complexity Scaling (True: {true_model})")
            ax.legend()
            ax.set_xscale('log')
            
            # Plot 2: Test error vs Rademacher complexity
            ax = axes[idx, 1]
            for model_type in data["test_model"].unique():
                model_data = data[data["test_model"] == model_type]
                ax.scatter(model_data["rademacher_complexity"],
                          model_data["test_error"],
                          alpha=0.5, label=model_type)
            ax.set_xlabel("Rademacher Complexity")
            ax.set_ylabel("Test Error")
            ax.set_title(f"Generalization vs Complexity (True: {true_model})")
            ax.legend()
            
            # Plot 3: Efficiency (complexity per parameter)
            ax = axes[idx, 2]
            for model_type in data["test_model"].unique():
                model_data = data[data["test_model"] == model_type]
                grouped = model_data.groupby("max_lag").mean(numeric_only=True)
                ax.plot(grouped.index,
                       grouped["complexity_per_param"],
                       marker='s', label=model_type)
            ax.set_xlabel("Max Lag")
            ax.set_ylabel("Complexity per Parameter")
            ax.set_title(f"Model Efficiency (True: {true_model})")
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def test_temporal_causality(self, forward_lag: int = 5, 
                               reverse_lag: int = 5) -> Dict:
        """
        Test whether imposing wrong causal direction increases complexity
        This tests NTW's claim that relatedness is consequence not cause
        
        Args:
            forward_lag: Lag for forward causality (cause → effect)
            reverse_lag: Lag for reverse causality (effect → cause)
        
        Returns:
            Dictionary with causality test results
        """
        # Generate data with known causal direction
        # Group formation (cause) → Eusociality (effect) → High relatedness (consequence)
        
        n_time = 200
        cause = np.random.randn(n_time)  # Ecological benefits
        effect = np.zeros(n_time)  # Eusociality emergence
        consequence = np.zeros(n_time)  # Relatedness
        
        # Forward causality with delay
        for t in range(forward_lag, n_time):
            effect[t] = 0.7 * cause[t - forward_lag] + 0.3 * np.random.randn()
        
        for t in range(forward_lag + reverse_lag, n_time):
            consequence[t] = 0.8 * effect[t - reverse_lag] + 0.2 * np.random.randn()
        
        # Test both causal directions
        results = {}
        
        # Correct direction: cause → effect
        X_correct = cause.reshape(-1, 1)
        y_correct = effect
        
        # Wrong direction: consequence → effect (forcing relatedness to be cause)
        X_wrong = consequence.reshape(-1, 1)
        y_wrong = effect
        
        analyzer = RademacherComplexityAnalyzer()
        
        # Measure complexity for correct causal model
        valid_points = forward_lag
        X_correct_valid = X_correct[:-valid_points]
        y_correct_valid = y_correct[valid_points:]
        
        complexity_correct = analyzer.compute_empirical_rademacher(
            X_correct_valid, y_correct_valid,
            Ridge, {"alpha": 0.01}
        )
        
        # Measure complexity for wrong causal model (needs more "epicycles")
        valid_points = forward_lag + reverse_lag
        X_wrong_valid = X_wrong[:-valid_points]
        y_wrong_valid = y_wrong[valid_points:]
        
        complexity_wrong = analyzer.compute_empirical_rademacher(
            X_wrong_valid, y_wrong_valid,
            Ridge, {"alpha": 0.01}
        )
        
        results["complexity_correct_direction"] = complexity_correct
        results["complexity_wrong_direction"] = complexity_wrong
        results["complexity_ratio"] = complexity_wrong / complexity_correct
        
        print("\n=== Temporal Causality Test ===")
        print(f"Complexity (correct: cause→effect): {complexity_correct:.4f}")
        print(f"Complexity (wrong: consequence→effect): {complexity_wrong:.4f}")
        print(f"Complexity ratio (wrong/correct): {results['complexity_ratio']:.2f}")
        
        if results["complexity_ratio"] > 1.5:
            print("✓ Result supports NTW: Wrong causal direction requires more complexity")
        else:
            print("✗ Result does not clearly support NTW's claim")
        
        return results

def main():
    """
    Run the complete experiment testing whether detailed kinship tracking
    improves generalization beyond simpler group-level models
    """
    
    print("="*60)
    print("Testing Rademacher Complexity of Temporal Causal Models")
    print("Comparing Kin Selection vs Group Selection")
    print("="*60)
    
    # Initialize experiment
    experiment = TemporalCausalityExperiment(
        n_individuals=50,  # Reduced for faster computation
        n_timepoints=300,
        n_groups=5
    )
    
    # Part 1: Compare model complexity and generalization
    print("\n1. Running complexity comparison experiments...")
    results_df = experiment.run_complexity_comparison(
        max_lags=[3, 5, 10],
        n_runs=5  # Reduced for faster demonstration
    )
    
    # Analyze results
    print("\n=== Summary Statistics ===")
    summary = results_df.groupby(["true_model", "test_model"]).agg({
        "rademacher_complexity": ["mean", "std"],
        "test_error": ["mean", "std"],
        "n_parameters": "mean"
    }).round(4)
    print(summary)
    
    # Check if simpler model generalizes as well
    for true_model in ["group", "kin"]:
        subset = results_df[results_df["true_model"] == true_model]
        
        kin_error = subset[subset["test_model"] == "kin_selection"]["test_error"].mean()
        group_error = subset[subset["test_model"] == "group_selection"]["test_error"].mean()
        
        kin_complexity = subset[subset["test_model"] == "kin_selection"]["rademacher_complexity"].mean()
        group_complexity = subset[subset["test_model"] == "group_selection"]["rademacher_complexity"].mean()
        
        print(f"\nTrue model: {true_model}")
        print(f"  Kin selection - Error: {kin_error:.4f}, Complexity: {kin_complexity:.4f}")
        print(f"  Group selection - Error: {group_error:.4f}, Complexity: {group_complexity:.4f}")
        
        if group_error < kin_error * 1.1 and group_complexity < kin_complexity:
            print(f"  ✓ Simpler group model generalizes as well with lower complexity")
        else:
            print(f"  ✗ Complex kin model shows advantage")
    
    # Part 2: Test temporal causality
    print("\n2. Testing temporal causality (NTW's claim)...")
    causality_results = experiment.test_temporal_causality()
    
    # Part 3: Visualize results
    print("\n3. Generating visualizations...")
    experiment.plot_results(results_df)
    
    # Final conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    # Calculate key metrics for conclusion
    group_data = results_df[results_df["true_model"] == "group"]
    kin_wins = group_data[group_data["test_model"] == "kin_selection"]["test_error"].mean()
    group_wins = group_data[group_data["test_model"] == "group_selection"]["test_error"].mean()
    
    if group_wins < kin_wins * 1.05:
        print("1. GROUP SELECTION SUFFICIENT: Simple group model generalizes")
        print("   as well as complex kin model for group selection dynamics")
    
    if causality_results["complexity_ratio"] > 1.5:
        print("2. CAUSALITY SUPPORTED: Forcing wrong causal direction")
        print("   (relatedness as cause) increases model complexity")
    
    # Efficiency analysis
    efficiency_ratio = (
        results_df[results_df["test_model"] == "group_selection"]["complexity_per_param"].mean() /
        results_df[results_df["test_model"] == "kin_selection"]["complexity_per_param"].mean()
    )
    
    print(f"3. EFFICIENCY: Group selection is {1/efficiency_ratio:.1f}x more")
    print(f"   parameter-efficient than kin selection")
    
    return results_df, causality_results

if __name__ == "__main__":
    results_df, causality_results = main()