import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from abm_simulation import generate_abm_data
from causal_discovery import CausalDiscovery
from scipy import stats

def run_robust_analysis():
    print("="*80)
    print("ROBUST CAUSAL ANALYSIS OF EUSOCIALITY EVOLUTION")
    print("Integrating Agent-Based Modeling, Causal Discovery, and Sensitivity Analysis")
    print("="*80)

    # 1. Generate Neutral Data from ABM
    print("\n1. DATA GENERATION (Agent-Based Model)")
    print("-" * 50)
    print("Simulating 500 independent evolutionary runs...")
    # Generate enough samples for statistical power
    df = generate_abm_data(n_samples=500)
    
    print("\nData Summary:")
    print(df.describe().round(3))
    
    # 2. Causal Discovery (Structure Learning)
    print("\n2. CAUSAL DISCOVERY (PC Algorithm)")
    print("-" * 50)
    print("Attempting to recover causal structure from observational data...")
    
    cd = CausalDiscovery(df, alpha=0.05)
    learned_graph = cd.run_pc_algorithm()
    
    print("\nLearned Edges:")
    if learned_graph.number_of_edges() == 0:
        print("  No edges found (variables might be independent or relationship is too weak/nonlinear)")
    else:
        for u, v in learned_graph.edges():
            print(f"  {u} -> {v}")
            
    # Check if NTW's structure was recovered
    # NTW claims: ecological -> colony -> eusociality
    # Hamilton claims: relatedness -> eusociality
    
    has_colony_eusocial = learned_graph.has_edge('colony_formation', 'eusociality')
    has_relatedness_eusocial = learned_graph.has_edge('relatedness', 'eusociality')
    
    print("\nStructure Validation:")
    if has_colony_eusocial:
        print("  [SUPPORT] Edge 'colony_formation -> eusociality' found.")
    else:
        print("  [MISSING] Edge 'colony_formation -> eusociality' NOT found.")
        
    if has_relatedness_eusocial:
        print("  [SUPPORT] Edge 'relatedness -> eusociality' found.")
    else:
        print("  [MISSING] Edge 'relatedness -> eusociality' NOT found.")
        
    # 3. Sensitivity Analysis (Rosenbaum Bounds)
    print("\n3. SENSITIVITY ANALYSIS (Hidden Confounders)")
    print("-" * 50)
    print("Testing robustness of 'Colony Formation -> Eusociality' link against hidden confounders.")
    
    # We want to know: How strong would a hidden confounder (U) have to be
    # to explain away the observed effect of Colony Formation on Eusociality?
    
    # Simplified Rosenbaum-style sensitivity analysis for continuous data
    # We compare the effect size with and without a hypothetical confounder
    
    # First, estimate naive effect size (correlation)
    r_naive = df['colony_formation'].corr(df['eusociality'])
    print(f"  Observed Correlation (Colony -> Eusociality): {r_naive:.4f}")
    
    # Gamma simulation
    # Gamma (Γ) represents the odds ratio of the confounder's effect on treatment assignment
    # Here we simulate how much the correlation would drop if we controlled for a strong confounder
    
    print("\n  Simulating hidden confounder U with varying strength...")
    print("  (Gamma represents the degree of hidden bias)")
    
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    robustness_limit = 1.0
    
    for gamma in gammas:
        # Approximate reduction in effect size due to confounding
        # This is a heuristic based on Rosenbaum's bounds logic
        # A confounder with odds ratio Gamma can reduce the t-statistic by a factor
        
        # In a rigorous analysis, we'd use matching, but for this demo we'll use
        # a simplified "tipping point" analysis.
        # How much of the variance in both X and Y would U need to explain?
        
        # Let's say U explains 'gamma_prop' of the variance
        # Gamma=1.0 means no confounding (0%)
        # Gamma=2.0 implies a strong confounder
        
        # We'll use the "Impact Threshold for a Confounding Variable" (ITCV) concept
        # ITCV = r_xy * r_uy
        
        # Calculate partial correlation assuming a confounder U exists
        # that correlates with both X and Y at level rho
        rho = (gamma - 1) / (gamma + 1) # Map Gamma [1, inf) to rho [0, 1)
        
        # Partial correlation formula: r_xy.u = (r_xy - r_xu*r_yu) / sqrt(...)
        # Worst case: confounder explains the correlation
        r_adjusted = (r_naive - rho*rho) / (1 - rho*rho)
        
        print(f"  Gamma = {gamma:.1f} (Confounder str ≈ {rho:.2f}): Adjusted Corr = {r_adjusted:.4f}")
        
        if r_adjusted <= 0.1: # Threshold for "no meaningful effect"
            robustness_limit = gamma
            break
            
    print(f"\n  Robustness Limit (Gamma): {robustness_limit}")
    if robustness_limit > 2.0:
        print("  -> ROBUST: It would take a very strong unobserved confounder (e.g., >2x odds)")
        print("     to explain away the colony formation effect.")
    else:
        print("  -> SENSITIVE: A moderate unobserved confounder could explain this result.")

    # 4. Instrumental Variable Analysis (Simulated)
    print("\n4. INSTRUMENTAL VARIABLE ANALYSIS")
    print("-" * 50)
    print("Using 'Ecological Pressure' as an instrument for Colony Formation.")
    
    # IV assumptions:
    # 1. Relevance: Z -> X (Eco Pressure -> Colony)
    # 2. Exclusion: Z -> Y only through X (Eco Pressure affects Eusociality ONLY via Colony)
    # 3. Independence: Z is independent of confounders
    
    # Stage 1: Regress X on Z
    # Colony = a + b*Eco + e
    slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(
        df['ecological_pressure'], df['colony_formation']
    )
    print(f"  Stage 1 (Relevance): Eco -> Colony (R²={r_value_1**2:.4f}, p={p_value_1:.4e})")
    
    if p_value_1 > 0.05:
        print("  [FAIL] Instrument is weak. Cannot proceed with IV analysis.")
    else:
        # Stage 2: Regress Y on predicted X
        predicted_colony = intercept_1 + slope_1 * df['ecological_pressure']
        
        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(
            predicted_colony, df['eusociality']
        )
        
        print(f"  Stage 2 (Causal Effect): Predicted Colony -> Eusociality")
        print(f"  IV Estimate of Causal Effect: {slope_2:.4f} (p={p_value_2:.4e})")
        print(f"  OLS Estimate (Naive): {stats.linregress(df['colony_formation'], df['eusociality'])[0]:.4f}")
        
        if abs(slope_2) > 0.1 and p_value_2 < 0.05:
            print("  [SUCCESS] IV analysis confirms causal link.")
        else:
            print("  [NULL] IV analysis suggests no causal link (association might be confounded).")

    print("\n" + "="*80)
    print("FINAL CONCLUSION")
    print("="*80)
    
    if has_colony_eusocial and robustness_limit > 1.5:
        print("The analysis STRONGLY SUPPORTS the NTW hypothesis:")
        print("1. Causal Discovery recovered the link from neutral ABM data.")
        print("2. The link is robust to moderate hidden confounding.")
        if p_value_1 < 0.05 and p_value_2 < 0.05:
            print("3. IV analysis confirms the causal direction.")
    elif has_relatedness_eusocial:
        print("The analysis SUPPORTS the Hamilton hypothesis:")
        print("1. Causal Discovery found a direct link from Relatedness to Eusociality.")
    else:
        print("The results are INCONCLUSIVE or support a mixed model.")

if __name__ == "__main__":
    import sys
    
    # Redirect stdout to both console and file
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open('pearl_results.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:
            run_robust_analysis()
        finally:
            sys.stdout = original_stdout
