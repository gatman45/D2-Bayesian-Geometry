# D2 — Bayesian Geometry of Weight Matrices

**Spectral Signatures of Criticality Across Biological and Artificial Networks**  
Extension empirique de Heavy-Tailed Self-Regularization (HT-SR)

**Ultra-simple et entièrement reproductible** — tout se lance en **2 commandes**.

## Résultats récents (S67 — mars 2026)

**Hypothèse principale validée à 100 %** sur 10/10 LLMs testés :

> **α_w(Attention) > α_w(MLP)** dans tous les modèles

- Δ moyen = **+0.245 ± 0.134**
- α_w(Attn) moyen = **0.684 ± 0.115**
- α_w(MLP) moyen = **0.365 ± 0.041**

**Différences par famille de modèles :**
- **Pythia** (3 modèles) : Δ = **+0.399 ± 0.059** → **2.25× plus fort** que GPT-2
- GPT-2 (4 modèles) : Δ = **+0.177 ± 0.026**
- Qwen, BLOOM, Neo : Δ entre +0.072 et +0.330

**Autres observations validées :**
- α_MLP ≤ 0.37 observé de manière quasi-universelle
- Différence statistiquement significative entre Pythia et GPT-2 (p=0.002)

![Résumé des résultats S67](results/s67_summary.png)
![Rapport détaillé S67](results/s67_detailed.png)

## Découverte principale
**Gap spectral Δα_w ≈ 0.98** entre les LLMs testés et des réseaux spiking neuronaux (SNN) biologiquement contraints à criticité.

## Quick Start

```bash
# 1. Installation des dépendances
pip install -r requirements.txt

# 2. Lancer l'analyse complète + génération du rapport
python d2_paper_generator.py
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
