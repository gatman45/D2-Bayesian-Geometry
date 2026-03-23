# D2 — Bayesian Geometry of Weight Matrices

> **Spectral Signatures of Criticality Across Biological and Artificial Networks**
> Validating and Extending Coppola et al. (2024)

---

## Summary

This repository contains all reproducible code, simulations, and results for the D2 paper.

We analyze the **empirical spectral density (ESD)** of weight matrices in:
- Biologically-constrained **Spiking Neural Networks (SNN)** at criticality
- Trained **Large Language Models** (GPT-2, DistilGPT-2)

### Key Findings

| Claim (Coppola) | Verdict |
|-----------------|---------|
| T16: Heavy-tailed spectra exist | ✅ Validated |
| T17: α_w varies between architectures | ✅ Validated |
| T18: Layer-wise quality correlation | ⚠️ Partially validated |

**New discovery:** A spectral gap Δα_w ≈ 0.98 between biological criticality and LLMs.

---

## Structure

```
D2-Bayesian-Geometry/
├── README.md                  # This file
├── d2_paper_generator.py      # Main script: runs all analyses
├── simulations/
│   ├── build_bio_W.py         # Builds biologically-constrained weight matrix
│   ├── simulate_lif.py        # LIF spiking neural network simulator
│   └── measure_alpha_w.py     # Spectral analysis pipeline
├── results/
│   └── alpha_w_results.json   # Pre-computed results (20 seeds)
├── docs/
│   └── D2_paper.md            # Full paper in Markdown
├── requirements.txt
└── LICENSE
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full analysis
python d2_paper_generator.py

# 3. Run individual modules
python simulations/build_bio_W.py
python simulations/simulate_lif.py
python simulations/measure_alpha_w.py
```

---

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib
- See `requirements.txt` for full list

---

## Reproducibility

All analyses use explicit random seeds (0–19).
Results are deterministic given fixed seeds.
Pre-computed results available in `results/alpha_w_results.json`.

---

## Theoretical Note

The constant β and its biological derivation are discussed in a companion paper
(in preparation). This repository does not expose or derive β — it is used
only as a reference value.

---

## Citation

```
[REDACTED] (2025). Bayesian Geometry of Weight Matrices:
Spectral Signatures of Criticality Across Biological and Artificial Networks.
Pre-print.
```

---

## License

MIT License — see LICENSE file.
