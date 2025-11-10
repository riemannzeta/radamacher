# **Rademacher Complexity Analysis of Temporal Causal Models**

This project implements a computational framework for comparing the Rademacher complexity and generalization performance of different temporal causal models in the context of eusocial evolution. It specifically tests whether detailed kinship tracking (kin selection) provides better predictive power than simpler group-level models (group selection). The project was developed to support an essay about [Radamacher Complexity and Models of Group Competition](https://www.google.com/search?q=https://www.symmetrybroken.com/group-selection/) published to the [Broken Symmetry](https://www.symmetrybroken.com/) blog.

## **Overview**

The code analyzes two competing models for explaining eusocial behavior:

1. **Kin Selection Model**: A complex model that tracks all pairwise relatedness coefficients across individuals with temporal lags. This represents the inclusive fitness approach with detailed kinship tracking.  
2. **Group Selection Model**: A simpler model that tracks only colony-level summary statistics (mean, variance, group size) across time. This represents a mean-field approximation that ignores detailed relatedness.

### **Key Features**

* **Rademacher Complexity Analysis**: Measures the empirical Rademacher complexity of both model classes to quantify their capacity to fit random noise  
* **Generalization Testing**: Compares test error between models on synthetic temporal data  
* **Temporal Causality Testing**: Tests whether imposing the wrong causal direction (treating relatedness as cause rather than consequence) increases model complexity  
* **Comparative Visualization**: Generates plots showing complexity scaling, generalization performance, and parameter efficiency

### **Theoretical Motivation**

This analysis addresses a key question in evolutionary biology: does tracking detailed genetic relatedness improve predictions about eusocial behavior, or are simpler group-level models sufficient? The Rademacher complexity framework provides a principled way to test whether additional model complexity translates to better generalization.

## **Installation**

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To install:

1. Install uv if you haven't already:

   ```bash
   curl \-LsSf \[https://astral.sh/uv/install.sh\](https://astral.sh/uv/install.sh) | sh
   ```

3. Clone the repository:

   ```bash
   git clone \[https://github.com/riemannzeta/radamacher.git\](https://github.com/riemannzeta/radamacher.git)  
   cd radamacher
   ```

5. Install dependencies:  

   ```bash
   uv sync
   ```

This will create a virtual environment and install all required dependencies specified in pyproject.toml:

* numpy  
* pandas  
* scipy  
* scikit-learn  
* matplotlib  
* tqdm  
* dataclasses  
* typing

## **Usage**

Run the complete analysis:

```bash
uv run python main.py
```

The script will:

1. Run complexity comparison experiments across different model configurations  
2. Test temporal causality by comparing correct vs. reversed causal directions  
3. Generate visualizations (displayed via matplotlib)  
4. Print summary statistics and conclusions to the console

### **Expected Output**

The analysis produces two types of output:

1. **Console Output** (results.txt): Contains:  
   * Summary statistics comparing Rademacher complexity and test error for both models  
   * Results for each true data-generating process (group vs. kin selection)  
   * Temporal causality test results  
   * Final conclusions about model efficiency and generalization  
2. **Visualization** (RadamacherComparison.png): A 2x3 grid of plots showing:  
   * **Left column**: Rademacher complexity vs. number of parameters (log scale)  
   * **Middle column**: Test error vs. Rademacher complexity (scatter plots)  
   * **Right column**: Complexity per parameter vs. maximum lag  
   * **Top row**: Results when true model is group selection  
   * **Bottom row**: Results when true model is kin selection

## **Output Examples**

### **RadamacherComparison.png**

This visualization compares the two models across multiple dimensions. Key insights:

* The kin selection model has orders of magnitude more parameters than the group selection model  
* Higher Rademacher complexity generally correlates with higher test error  
* The group selection model shows better parameter efficiency (lower complexity per parameter)

### **results.txt**

Example output showing that the simpler group selection model generalizes as well as the complex kin selection model, with a temporal causality test confirming that forcing the wrong causal direction increases complexity (supporting the claim that relatedness is a consequence, not a cause, of eusociality).

## **Project Structure**

```
radamacher/  
├── main.py                 \# Main analysis script  
├── pyproject.toml          \# Project dependencies and metadata  
├── uv.lock                 \# Locked dependency versions  
├── README.md               \# This file  
├── comparison.png          \# Example output visualization  
└── results.txt             \# Example console output (not included in repo)
```

## **Scientific Context**

This work relates to debates about the evolution of eusociality:

* **Inclusive fitness theory** emphasizes individual-level selection weighted by genetic relatedness  
* **Multilevel selection theory** emphasizes selection operating at multiple hierarchical levels

By comparing model complexity and generalization, this analysis provides a computational lens on which framework offers better predictive power for a given system.

## **Customization**

You can modify the experiment parameters in main() (main.py:707):

```python
experiment \= TemporalCausalityExperiment(
    n\_individuals=50,     \# Number of individuals in the population
    n\_timepoints=300,     \# Length of time series
    n\_groups=5            \# Number of groups/colonies
)

results\_df \= experiment.run\_complexity\_comparison(
    max\_lags=\[3, 5, 10\],  \# Different temporal lag lengths to test
    n\_runs=5              \# Number of experimental replications
)
```

## **License**

MIT

## **Citation**

If you use this code in your research, please cite appropriately.
