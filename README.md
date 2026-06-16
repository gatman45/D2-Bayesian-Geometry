# D2 — Bayesian Geometry of Weight Matrices

**Spectral Signatures of Criticality Across Biological and Artificial Networks**

Reproducible analysis of spectral properties in neural network weight matrices.  
Connects biological criticality (SNN) to deep learning weight structure (LLMs).

> **Compatibility**: drop-in complement to [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) — adds pseudospectral analysis (Henrici, Kreiss) not available elsewhere.

---

## What This Project Does

Three concrete things:

1. **Measures α_w** (spectral exponent) on any 2D weight matrix — SNN or LLM
2. **Validates** three spectral hypotheses from Coppola (arXiv:2603.17063) on biological SNN baselines
3. **Extends** the analysis with non-normality / pseudospectral diagnostics (Henrici measure, Kreiss constant)

---

## Key Results

### Measured in this project (10 seeds, N=200)

```
SNN at criticality (σ_B = 1.04):
  α_w = 2.1732 ± 0.0939
  Range: [2.0934, 2.3421]
  Branching parameter m ≈ 0.98  (expected ≈ 1.0)
```

### Replicated from Coppola (arXiv:2603.17063)

```
GPT-2 (124M)      α_w = 3.08
DistilGPT-2 (82M) α_w = 3.82
Pythia family     α_w = 3.14

Spectral gap (SNN → GPT-2): Δα_w = 0.91  (~42% relative)
```

> **Note**: LLM α_w values above are sourced from Coppola et al., not independently re-measured here.  
> Independent re-measurement on arbitrary HuggingFace models: see [Advanced Usage](#advanced-usage).

### Hypothesis validation

| Hypothesis | Prediction | Result | Source |
|---|---|---|---|
| H1: Heavy tails | α_w < 4.0 | SNN: 2.17, GPT-2: 3.08 ✅ | This work + Coppola |
| H2: Architecture variance | Δα_w > 0.3 across models | GPT-2 vs DistilGPT-2: Δ=0.74 ✅ | Coppola |
| H3: Attn < MLP | α_w(Attn) < α_w(MLP) | 10/10 LLMs ✅ | Coppola |

---

## Quick Start

```bash
pip install -r requirements.txt

# All-in-one: theory + code + validation
python D2_COMPLETE_GUIDE.py

# Output: results/D2_complete_analysis.json
```

### Measure α_w on any HuggingFace model

```python
from transformers import AutoModelForCausalLM
from simulations.measure_alpha_w import measure_alpha_w
import torch, numpy as np

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float32   # cast required — model loads in bfloat16 by default
)

results = {}
for name, param in model.named_parameters():
    W = param.detach().cpu().float().numpy()
    if W.ndim == 2 and min(W.shape) >= 32:
        alpha_w = measure_alpha_w(W)
        layer_type = (
            "attn" if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
            else "mlp"
        )
        results[name] = {"shape": W.shape, "alpha_w": alpha_w, "type": layer_type}

attn = [v["alpha_w"] for v in results.values() if v["type"] == "attn"]
mlp  = [v["alpha_w"] for v in results.values() if v["type"] == "mlp"]
print(f"α_w Attn : {np.mean(attn):.4f} ± {np.std(attn):.4f}")
print(f"α_w MLP  : {np.mean(mlp):.4f}  ± {np.std(mlp):.4f}")
print(f"H3 (Attn < MLP): {np.mean(attn) < np.mean(mlp)}")
```

> **GQA note (Qwen3, Mistral, Llama-3)**: K/V projections are smaller matrices under GQA.  
> Stratify `attn_kv` separately from `q_proj`/`o_proj` for accurate layer-type comparison.

---

## Project Structure

```
D2-Bayesian-Geometry/
├── D2_COMPLETE_GUIDE.py          # Entry point — run this first
├── d2_paper_generator.py         # Full pipeline (20 seeds)
├── run_real_test.py              # Test suite
│
├── simulations/
│   ├── build_bio_W.py            # Biologically-constrained weight matrices
│   ├── measure_alpha_w.py        # SVD → log-log → α_w
│   └── simulate_lif.py           # LIF spiking network + avalanche detection
│
├── results/
│   ├── D2_complete_analysis.json
│   └── alpha_w_results.json
│
└── docs/
    └── D2_paper.md               # Full theory
```

---

## Scientific Background

### Heavy-Tailed Self-Regularization (HT-SR)

Singular values of a weight matrix W ∈ ℝ^{N×N} follow:

```
σ_i ~ C × i^{-α_w/2}

Measurement:
  log(σ_i) = intercept + slope × log(i)
  α_w = -2 × slope

Interpretation:
  α_w < 4  → heavy tails (good generalization, per Martin & Mahoney 2021)
  α_w > 5  → thin tails (poor generalization)
```

### Criticality

Biological networks operate at the edge of chaos (σ_B ≈ 1.04):

```
At σ_B = 1.04:
  Branching parameter m → 1.0   (measured here: m ≈ 0.98 ✅)
  Lyapunov exponent λ → 0
  Maximum information capacity
```

### Non-Normal Amplification (extended analysis)

Beyond α_w, this project characterizes **transient dynamics** via:

- **Henrici measure**: `||W^T W - W W^T||_F / ||W||_F²`
- **Kreiss constant**: `K = sup_{ε>0} ε × sup_{|z|=1+ε} ||(zI-W)^{-1}||`
- **Pseudospectrum**: contour-integrated resolvent (not max norm — see limitations)

**Core finding**: non-normality does NOT shift the asymptotic phase transition (g_c = 1/ρ(W) holds).  
It creates transient amplification (~+5%) and distorts early-warning signals — without moving the critical point.

---

## Known Limitations

| ID | Issue | Impact |
|---|---|---|
| L1 | n < 30 effective crossings in several experiments | finite-size noise in g_c estimates |
| L2 | Naive max-resolvent diverges — contour integration required | pseudospectrum values not comparable across papers using max-norm |
| L3 | FTLE window T too small in some runs | underestimates Lyapunov fluctuations |
| L4 | Non-normality not fully orthogonalized vs ρ(W) in early runs | γ dependence confound (resolved in final pipeline) |
| L5 | LLM α_w values from Coppola — not re-measured independently | cannot claim exact replication without re-running on actual model weights |

---

## Advanced Usage

### Explore σ_B effect on α_w

```python
from simulations.build_bio_W import build_weight_matrix
from simulations.measure_alpha_w import measure_alpha_w

for sigma_B in [0.99, 1.00, 1.04, 1.10]:
    W = build_weight_matrix(N=500, sigma_B=sigma_B, seed=0)
    print(f"σ_B={sigma_B}: α_w={measure_alpha_w(W):.4f}")
```

### Fast mode (5 minutes)

```bash
python d2_paper_generator.py --fast --seeds 5
```

### Reduced memory

```bash
python d2_paper_generator.py --network-size 100 --seeds 5
```

---

## Reproducibility

All experiments use explicit seeds (0–19):

```python
W1 = build_weight_matrix(N=500, sigma_B=1.04, seed=0)
W2 = build_weight_matrix(N=500, sigma_B=1.04, seed=0)
assert np.allclose(W1, W2)  # True
```

Pre-computed results: `results/D2_complete_analysis.json`, `results/alpha_w_results.json`

---

## References

- **Coppola, G. (2026)**. arXiv:2603.17063. *(primary theoretical foundation — H1/H2/H3 sourced here)*
- **Martin, C.H. & Mahoney, M.W. (2021)**. "Implicit Self-Regularization in Deep Neural Networks". JMLR. *(HT-SR framework)*
- **Beggs, J.M. & Plenz, D. (2003)**. "Neuronal Avalanches in Neocortical Circuits". *Journal of Neuroscience*. *(criticality baseline)*
- **Henrici, P. (1962)**. *Bounds for Iterates, Inverses, Spectral Variation and Fields of Values of Non-Normal Matrices*. *(non-normality measure)*

---

## Relation to WeightWatcher

[WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) (Martin & Mahoney) measures α_w on pretrained models via `pip install weightwatcher`.

**D2 adds**:
- Biological SNN baseline (σ_B sweep, branching parameter validation)
- Non-normality / pseudospectral layer (Henrici, Kreiss) — not in WeightWatcher
- Explicit criticality framing (DMFT, g_c invariance)

**D2 does not replace** WeightWatcher for production LLM auditing.

---

## Citation

```bibtex
@software{d2_bayesian_2026,
  title   = {D2: Bayesian Geometry of Weight Matrices},
  author  = {JARVIX Research Group},
  year    = {2026},
  url     = {https://github.com/gatman45/D2-Bayesian-Geometry},
  note    = {Spectral and pseudospectral analysis of criticality in SNN and LLM weight matrices}
}
```

---

## License

MIT
