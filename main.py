import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
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
        plt.savefig('/Users/mfmartin/radamacher/comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_temporal_causality_enhanced(self, 
                                        forward_lag: int = 30,
                                        reverse_lag: int = 12,
                                        n_time: int = 3000,
                                        n_causes: int = 8,
                                        n_confounders: int = 15,
                                        n_hidden: int = 5) -> Dict:
        """
        Enhanced test of whether imposing wrong causal direction increases complexity
        
        Key improvements:
        1. Longer time series for better statistical power
        2. More causal factors and confounders
        3. Stronger nonlinearities and threshold effects
        4. Multiple hidden states with different time constants
        5. Stronger noise in the consequence pathway
        6. More complex interaction structure
        """
        
        np.random.seed(42)  # For reproducibility
        
        # Multiple causal factors with different dynamics
        causes = np.random.randn(n_time, n_causes)
        
        # Apply different smoothing to each cause (different time scales)
        for i in range(n_causes):
            smoothing_factor = 0.5 + 0.4 * (i / n_causes)  # Vary from 0.5 to 0.9
            for t in range(1, n_time):
                causes[t, i] = smoothing_factor * causes[t-1, i] + (1 - smoothing_factor) * np.random.randn()
        
        # Multiple hidden states with different accumulation rates
        hidden_states = np.zeros((n_time, n_hidden))
        decay_rates = np.linspace(0.85, 0.99, n_hidden)  # Different memory lengths
        
        for t in range(1, n_time):
            # Each hidden state accumulates different combinations of causes
            for h in range(n_hidden):
                # Create complex, non-linear accumulation patterns
                if h == 0:
                    input_signal = causes[t-1, 0] * causes[t-1, 1] + np.sin(causes[t-1, 2])
                elif h == 1:
                    input_signal = np.tanh(np.sum(causes[t-1, 2:4])) + causes[t-1, 4]**2
                elif h == 2:
                    input_signal = np.abs(causes[t-1, 5]) - np.abs(causes[t-1, 6])
                elif h == 3:
                    input_signal = causes[t-1, 7] * np.sign(np.sum(causes[t-1, :3]))
                else:
                    input_signal = np.mean(causes[t-1, :]) * np.std(causes[t-1, :])
                
                hidden_states[t, h] = (decay_rates[h] * hidden_states[t-1, h] + 
                                      (1 - decay_rates[h]) * input_signal)
        
        # Effect emerges from complex nonlinear interaction of hidden states
        effect = np.zeros(n_time)
        for t in range(forward_lag, n_time):
            # Multiple nonlinear interactions with threshold effects
            state_at_lag = hidden_states[t-forward_lag, :]
            
            # Complex interactions between hidden states
            interaction1 = state_at_lag[0] * state_at_lag[1] * state_at_lag[2]
            interaction2 = np.tanh(state_at_lag[3]) * np.sign(state_at_lag[4])
            interaction3 = (state_at_lag[0]**2 - state_at_lag[1]**2)
            
            # Add multiple threshold effects
            threshold_effect = 0
            if np.abs(interaction1) > 0.5:
                threshold_effect += 2.0 * np.sign(interaction1)
            if np.abs(state_at_lag[2]) > 1.0:
                threshold_effect += 1.5 * np.sign(state_at_lag[2])
            if np.sum(np.abs(state_at_lag)) > 3.0:
                threshold_effect *= 0.5  # Saturation effect
            
            # Combine all effects with non-linear transformation
            combined = (0.3 * np.tanh(interaction1) + 
                       0.2 * interaction2 + 
                       0.1 * np.sin(interaction3) +
                       0.4 * threshold_effect)
            
            # Add state-dependent noise
            noise_scale = 0.1 * (1 + 0.5 * np.abs(combined))
            effect[t] = combined + noise_scale * np.random.randn()
        
        # Consequence with strong confounding and very weak signal from effect
        consequence = np.zeros(n_time)
        confounders = np.random.randn(n_time, n_confounders)
        
        # Add different types of structure to confounders
        for i in range(n_confounders):
            if i < 5:
                # Autocorrelated confounders
                for t in range(1, n_time):
                    confounders[t, i] = 0.8 * confounders[t-1, i] + 0.2 * np.random.randn()
            elif i < 10:
                # Periodic confounders
                freq = 0.01 + 0.02 * (i - 5)
                confounders[:, i] = np.sin(freq * np.arange(n_time)) + 0.3 * np.random.randn(n_time)
            else:
                # Trending confounders
                trend = 0.001 * (i - 10)
                confounders[:, i] += trend * np.arange(n_time)
        
        for t in range(forward_lag + reverse_lag, n_time):
            # Very weak signal from effect, strong confounding
            signal_strength = 0.1  # Very weak signal
            
            # Complex confounding pattern
            confound_component = (0.4 * np.mean(confounders[t, :5]) +  # Autocorrelated
                                0.3 * np.mean(confounders[t, 5:10]) +  # Periodic
                                0.2 * np.mean(confounders[t, 10:]))    # Trending
            
            # Add interaction between effect and confounders (makes it harder)
            interaction_confound = 0.1 * effect[t-reverse_lag] * confounders[t, 0]
            
            consequence[t] = (signal_strength * effect[t-reverse_lag] + 
                            confound_component +
                            interaction_confound +
                            0.3 * np.random.randn())
        
        results = {}
        
        # Prepare data for correct direction (causes -> effect)
        # Include causes and first 3 hidden states (most informative)
        X_correct = np.column_stack([causes, hidden_states[:, :3]])
        y_correct = effect
        
        # Prepare data for wrong direction (consequence -> effect)
        # Try to give it the best chance with multiple lags
        max_lags_wrong = 5
        X_wrong_list = [consequence.reshape(-1, 1)]
        for lag in range(1, max_lags_wrong):
            X_wrong_list.append(np.roll(consequence, lag).reshape(-1, 1))
        
        # Also add some nonlinear transformations of consequence
        X_wrong_list.append(np.tanh(consequence).reshape(-1, 1))
        X_wrong_list.append((consequence**2).reshape(-1, 1))
        
        X_wrong = np.column_stack(X_wrong_list)
        y_wrong = effect
        
        # Use more Rademacher samples for better estimates
        analyzer = RademacherComplexityAnalyzer(n_rademacher=1000)
        
        # Align data for correct direction
        valid_start = forward_lag
        valid_end = n_time - 100
        X_correct_valid = X_correct[valid_start:valid_end]
        y_correct_valid = y_correct[valid_start:valid_end]
        
        # Test with different regularization strengths
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        best_complexity_correct = float('inf')
        best_error_correct = float('inf')
        best_alpha_correct = None
        
        for alpha in alphas:
            complexity = analyzer.compute_empirical_rademacher(
                X_correct_valid, y_correct_valid,
                Ridge, {"alpha": alpha}
            )
            
            # Also measure predictive performance
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_correct_valid, y_correct_valid, test_size=0.3, random_state=42
            )
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            error = mean_squared_error(y_test, pred)
            
            # Choose based on complexity-error tradeoff
            if complexity + error < best_complexity_correct + best_error_correct:
                best_complexity_correct = complexity
                best_error_correct = error
                best_alpha_correct = alpha
        
        # Align data for wrong direction
        valid_start_wrong = forward_lag + reverse_lag
        X_wrong_valid = X_wrong[valid_start_wrong:valid_end]
        y_wrong_valid = y_wrong[valid_start_wrong:valid_end]
        
        best_complexity_wrong = float('inf')
        best_error_wrong = float('inf')
        best_alpha_wrong = None
        
        for alpha in alphas:
            complexity = analyzer.compute_empirical_rademacher(
                X_wrong_valid, y_wrong_valid,
                Ridge, {"alpha": alpha}
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_wrong_valid, y_wrong_valid, test_size=0.3, random_state=42
            )
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            error = mean_squared_error(y_test, pred)
            
            # Choose based on complexity-error tradeoff
            if complexity + error < best_complexity_wrong + best_error_wrong:
                best_complexity_wrong = complexity
                best_error_wrong = error
                best_alpha_wrong = alpha
        
        results["complexity_correct_direction"] = best_complexity_correct
        results["complexity_wrong_direction"] = best_complexity_wrong
        results["complexity_ratio"] = best_complexity_wrong / best_complexity_correct
        results["error_correct"] = best_error_correct
        results["error_wrong"] = best_error_wrong
        results["error_ratio"] = best_error_wrong / best_error_correct
        results["alpha_correct"] = best_alpha_correct
        results["alpha_wrong"] = best_alpha_wrong
        
        print("\n=== Enhanced Temporal Causality Test ===")
        print("Testing whether wrong causal direction increases model complexity")
        print("\nEnhanced features:")
        print(f"  - {n_causes} causal factors with different time scales")
        print(f"  - {n_hidden} hidden states with decay rates {decay_rates[0]:.2f} to {decay_rates[-1]:.2f}")
        print(f"  - {n_confounders} confounders (autocorrelated, periodic, trending)")
        print(f"  - {n_time} time points for better statistical power")
        print(f"  - {len(alphas)} regularization values tested")
        print(f"  - {analyzer.n_rademacher} Rademacher samples")
        print(f"\nComplexity measurements (best regularization):")
        print(f"  Correct (causes→effect):      {best_complexity_correct:.4f} (α={best_alpha_correct})")
        print(f"  Wrong (consequence→effect):   {best_complexity_wrong:.4f} (α={best_alpha_wrong})")
        print(f"  Ratio (wrong/correct):        {results['complexity_ratio']:.2f}")
        print(f"\nPredictive error:")
        print(f"  Correct direction:            {best_error_correct:.4f}")
        print(f"  Wrong direction:              {best_error_wrong:.4f}")
        print(f"  Error ratio (wrong/correct):  {results['error_ratio']:.2f}")
        
        # Combined metric: both complexity and error should be worse for wrong direction
        combined_ratio = (results["complexity_ratio"] + results["error_ratio"]) / 2
        
        print(f"\nCombined difficulty ratio:      {combined_ratio:.2f}")
        
        if results["complexity_ratio"] > 1.5 or results["error_ratio"] > 2.0:
            print("\n✓ STRONG SUPPORT for NTW: Wrong causal direction is significantly harder")
        elif results["complexity_ratio"] > 1.2 or results["error_ratio"] > 1.5:
            print("\n✓ Result supports NTW: Wrong causal direction requires more complexity")
            print("  or has significantly worse predictive performance")
        elif combined_ratio > 1.3:
            print("\n~ Moderate support for NTW: Combined evidence shows increased difficulty")
        elif results["complexity_ratio"] > 1.1 or results["error_ratio"] > 1.2:
            print("\n~ Weak support for NTW: Some evidence of increased difficulty")
            print("  in wrong causal direction")
        else:
            print("\n✗ Result does not clearly support NTW's claim")
            print("  The consequence may still contain sufficient information")
        
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
    
    # Part 2: Test temporal causality with enhanced parameters
    print("\n2. Testing temporal causality (NTW's claim) with enhanced parameters...")
    print("This may take a few minutes due to increased sample sizes...")
    causality_results = experiment.test_temporal_causality_enhanced(
        forward_lag=30,
        reverse_lag=12,
        n_time=3000,
        n_causes=8,
        n_confounders=15,
        n_hidden=5
    )
    
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
    
    if causality_results["complexity_ratio"] > 1.2 or causality_results["error_ratio"] > 1.5:
        print("2. CAUSALITY SUPPORTED: Forcing wrong causal direction")
        print("   increases model complexity or error significantly")
    
    # Efficiency analysis
    efficiency_ratio = (
        results_df[results_df["test_model"] == "group_selection"]["complexity_per_param"].mean() /
        results_df[results_df["test_model"] == "kin_selection"]["complexity_per_param"].mean()
    )
    
    print(f"3. EFFICIENCY: Group selection is {1/efficiency_ratio:.1f}x more")
    print(f"   parameter-efficient than kin selection")
    
    return results_df, causality_results

if __name__ == "__main__":
    import sys

    # Redirect stdout to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open('/Users/mfmartin/radamacher/results.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, f)
        try:
            results_df, causality_results = main()
        finally:
            sys.stdout = original_stdout