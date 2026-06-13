# HOW TO USE THIS PROJECT

## Quick Start (5 minutes)

### Option 1: Complete Guide (Recommended)

```bash
# Install
pip install -r requirements.txt

# Run everything
python D2_COMPLETE_GUIDE.py

# Output: results/D2_complete_analysis.json
```

This single command gives you:
- ✅ All theory explained
- ✅ Working code examples
- ✅ Real results (SNN vs LLM comparison)
- ✅ Validation of theorems

### Option 2: Step-by-Step

```bash
# Step 1: Build weight matrices
python -c "from simulations.build_bio_W import build_weight_matrix; W = build_weight_matrix(100); print(W.shape)"

# Step 2: Measure spectral exponent
python -c "from simulations.measure_alpha_w import measure_alpha_w; print(f'alpha_w = {measure_alpha_w(W):.4f}')"

# Step 3: Run full analysis
python simulations/measure_alpha_w.py
```

### Option 3: Interactive Jupyter

```bash
jupyter notebook

# Create new cell
from simulations.build_bio_W import build_weight_matrix
from simulations.measure_alpha_w import measure_alpha_w

W = build_weight_matrix(N=200, sigma_B=1.04, seed=0)
alpha_w = measure_alpha_w(W)
print(f'alpha_w = {alpha_w:.4f}')
```

---

## Understanding the Code

### 1. Build Weight Matrix

**What it does**: Creates a sparse, random weight matrix with spectral radius σ_B ≈ 1.04 (criticality).

**Why**: Mimics biological neural networks at edge of chaos.

```python
from simulations.build_bio_W import build_weight_matrix

W = build_weight_matrix(N=500, sigma_B=1.04, seed=0)
# W is (500, 500) matrix
# Sparse: ~10% non-zero
# Spectral radius: ~1.04
```

### 2. Measure Alpha-W

**What it does**: Computes the power-law exponent α_w from singular value spectrum.

**Why**: Characterizes "quality" of weight matrix for learning.

```python
from simulations.measure_alpha_w import measure_alpha_w

alpha_w = measure_alpha_w(W)
# alpha_w ≈ 2.17 for biological networks
# alpha_w ≈ 3.08 for GPT-2
```

### 3. Run LIF Simulation

**What it does**: Simulates spiking neural network dynamics.

**Why**: Validates that σ_B=1.04 produces critical avalanche dynamics (m ≈ 1.0).

```python
from simulations.simulate_lif import simulate_lif_network, compute_branching_parameter

results = simulate_lif_network(W, T=1.0, seed=0)
m = compute_branching_parameter(results['avalanches'])
# m ≈ 0.98 (should be ≈ 1.0 at criticality)
```

### 4. Full Analysis Pipeline

**What it does**: Runs 20 random seeds, computes statistics, compares with LLMs.

**Why**: Provides robust results + validation of Coppola theorems.

```python
from simulations.measure_alpha_w import run_full_analysis

summary = run_full_analysis(N=500, n_seeds=20, sigma_B=1.04)
# summary['mean_alpha_w'] ≈ 2.173
# summary['std_alpha_w'] ≈ 0.094
```

---

## Common Tasks

### Task 1: Reproduce the Paper Results

```bash
python D2_COMPLETE_GUIDE.py
```

**Output**: `results/D2_complete_analysis.json`

**Contains**:
- SNN spectral analysis (20 seeds)
- LLM reference values
- Theorem validation
- Spectral gap computation

### Task 2: Measure Alpha-W for Custom Matrix

```python
import numpy as np
from simulations.measure_alpha_w import measure_alpha_w

# Your custom matrix
W_custom = np.random.randn(100, 100) * 0.1

# Measure
alpha_w = measure_alpha_w(W_custom)
print(f'alpha_w = {alpha_w:.4f}')
```

### Task 3: Study Parameter Dependence

```python
from simulations.build_bio_W import build_weight_matrix
from simulations.measure_alpha_w import measure_alpha_w
import numpy as np

# Vary network size
results = {}
for N in [50, 100, 200, 500]:
    W = build_weight_matrix(N=N, sigma_B=1.04, seed=0)
    alpha_w = measure_alpha_w(W)
    results[N] = alpha_w
    print(f'N={N}: alpha_w={alpha_w:.4f}')

# Vary spectral radius
for sigma_B in [0.99, 1.00, 1.04, 1.10]:
    W = build_weight_matrix(N=200, sigma_B=sigma_B, seed=0)
    alpha_w = measure_alpha_w(W)
    print(f'sigma_B={sigma_B}: alpha_w={alpha_w:.4f}')
```

### Task 4: Batch Processing

```bash
# Run analysis for seeds 0-19
for seed in {0..19}; do
    python -c "
from simulations.build_bio_W import build_weight_matrix
from simulations.measure_alpha_w import measure_alpha_w
W = build_weight_matrix(N=500, sigma_B=1.04, seed=$seed)
alpha_w = measure_alpha_w(W)
print(f'Seed $seed: alpha_w={alpha_w:.4f}')
    " >> results/alpha_w_all_seeds.txt
done

cat results/alpha_w_all_seeds.txt
```

### Task 5: Compare with Your LLM

```python
import numpy as np
from simulations.measure_alpha_w import measure_alpha_w

# Load your model's weight matrix
# (example: from PyTorch)
# model = torch.load('my_model.pth')
# W = model.layer1.weight.data.numpy()

# Or simulate a matrix
W = np.random.randn(1000, 1000) * 0.01

alpha_w = measure_alpha_w(W)

print(f"Your model:  α_w = {alpha_w:.4f}")
print(f"GPT-2:       α_w = 3.08")
print(f"SNN optimal: α_w = 2.17")
print(f"Gap:         Δ = {alpha_w - 2.17:.4f}")
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'simulations'"

**Solution**: Make sure you're in the repo root

```bash
cd D2-Bayesian-Geometry
python D2_COMPLETE_GUIDE.py
```

### Problem: "MemoryError: Unable to allocate X GiB"

**Solution**: Reduce network size

```bash
python d2_paper_generator.py --network-size 100 --seeds 5
```

### Problem: Results don't match README

**Reason**: Stochastic variation. Need more seeds.

**Solution**:

```bash
python d2_paper_generator.py --seeds 50  # Instead of 20
```

### Problem: "ImportError: No module named scipy"

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Advanced Usage

### Visualize Results

```python
import matplotlib.pyplot as plt
import json

# Load results
with open('results/D2_complete_analysis.json') as f:
    data = json.load(f)

alpha_w_vals = data['snn_analysis']['alpha_w_values']

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(alpha_w_vals, bins=10, edgecolor='black')
plt.axvline(2.17, color='red', linestyle='--', label='Mean')
plt.xlabel('α_w')
plt.ylabel('Count')
plt.legend()
plt.savefig('results/alpha_w_distribution.png')
plt.show()
```

### Custom Analysis Script

```python
# my_analysis.py
import sys
sys.path.insert(0, '.')

from simulations.build_bio_W import build_weight_matrix
from simulations.measure_alpha_w import measure_alpha_w
from simulations.simulate_lif import simulate_lif_network, compute_branching_parameter
import numpy as np

def my_experiment():
    """Custom experiment"""
    
    print("Running custom analysis...")
    
    # Vary both parameters
    results = []
    for N in [100, 200, 500]:
        for sigma_B in [1.00, 1.04, 1.08]:
            W = build_weight_matrix(N=N, sigma_B=sigma_B, seed=0)
            alpha_w = measure_alpha_w(W)
            
            # Also simulate
            sim = simulate_lif_network(W, T=0.5, seed=0)
            m = compute_branching_parameter(sim['avalanches']) if sim['avalanches'] else 0
            
            results.append({
                'N': N,
                'sigma_B': sigma_B,
                'alpha_w': alpha_w,
                'branching_param': m
            })
            
            print(f"N={N}, σ_B={sigma_B}: α_w={alpha_w:.4f}, m={m:.4f}")
    
    return results

if __name__ == '__main__':
    results = my_experiment()
```

Run with:
```bash
python my_analysis.py
```

---

## Performance Tips

### Speed Up Analysis

```python
# Slow: N=1000, n_seeds=100
run_full_analysis(N=1000, n_seeds=100)

# Fast: N=200, n_seeds=5
run_full_analysis(N=200, n_seeds=5)
```

### Reduce Memory

```python
# Slow: Load all seeds at once
W_all = [build_weight_matrix(N=1000, seed=i) for i in range(20)]
alpha_all = [measure_alpha_w(W) for W in W_all]

# Fast: Process one at a time
for seed in range(20):
    W = build_weight_matrix(N=1000, seed=seed)
    alpha = measure_alpha_w(W)
    # Process immediately
```

### Parallel Processing

```python
from multiprocessing import Pool
from simulations.build_bio_W import build_weight_matrix
from simulations.measure_alpha_w import measure_alpha_w

def compute_alpha(seed):
    W = build_weight_matrix(N=500, seed=seed)
    return measure_alpha_w(W)

if __name__ == '__main__':
    with Pool(4) as p:
        results = p.map(compute_alpha, range(20))
    
    print(f"Mean: {np.mean(results):.4f}")
    print(f"Std:  {np.std(results):.4f}")
```

---

## Next Steps

After running this project:

1. **Understand the theory**: Read `docs/D2_THEORY.md`
2. **Explore variations**: Modify parameters and observe results
3. **Test on your data**: Load your own weight matrices
4. **Contribute**: Share improvements or findings!

---

## Questions?

Check:
- `README.md` — Overview
- `docs/D2_THEORY.md` — Theoretical details
- `D2_COMPLETE_GUIDE.py` — Detailed walkthrough
- `docs/HOWTO.md` — This file
