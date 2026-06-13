"""
measure_alpha_w.py
==================
Measure the spectral exponent alpha_w (power-law exponent of singular values).

Heavy-tailed Self-Regularization (HT-SR): alpha_w < 4 indicates heavy tails.
"""

import json
import numpy as np
from scipy.linalg import svd
from typing import Dict, Any
from .build_bio_W import build_weight_matrix


def measure_alpha_w(W: np.ndarray, min_singular_value: float = 1e-5) -> float:
    """
    Measure the power-law exponent alpha_w from singular value spectrum.
    
    Uses log-log regression on singular values to estimate:
        σ_i ~ i^(-alpha_w / 2)
    
    Args:
        W: Weight matrix (N, N)
        min_singular_value: Threshold for including singular values
    
    Returns:
        alpha_w: Estimated spectral exponent
    """
    U, S, Vt = svd(W, full_matrices=False)
    
    # Filter out near-zero singular values
    S_filtered = S[S > min_singular_value]
    
    if len(S_filtered) < 2:
        return 2.0  # Default fallback
    
    # Log-log regression: log(σ_i) ~ -alpha_w/2 * log(i)
    indices = np.arange(1, len(S_filtered) + 1)
    log_indices = np.log(indices)
    log_singular = np.log(S_filtered)
    
    # Linear regression
    coeffs = np.polyfit(log_indices, log_singular, 1)
    slope = coeffs[0]
    alpha_w = -2 * slope
    
    return float(max(alpha_w, 1.0))  # Ensure positive


def run_full_analysis(N: int = 500, n_seeds: int = 20, sigma_B: float = 1.04,
                      output_path: str = "results/alpha_w_results.json") -> Dict[str, Any]:
    """
    Run spectral analysis across multiple random seeds.
    
    Args:
        N: Network size
        n_seeds: Number of random seeds
        sigma_B: Target spectral radius
        output_path: Path to save results JSON
    
    Returns:
        Dictionary with summary statistics
    """
    alpha_w_values = []
    
    print(f"  Measuring alpha_w across {n_seeds} seeds...")
    for seed in range(n_seeds):
        W = build_weight_matrix(N=N, sigma_B=sigma_B, seed=seed)
        alpha_w = measure_alpha_w(W)
        alpha_w_values.append(float(alpha_w))
        print(f"    Seed {seed:2d}: alpha_w = {alpha_w:.4f}")
    
    alpha_w_values = np.array(alpha_w_values)
    
    summary = {
        "n_seeds": n_seeds,
        "network_size": N,
        "sigma_B": sigma_B,
        "alpha_w_values": alpha_w_values.tolist(),
        "mean_alpha_w": float(np.mean(alpha_w_values)),
        "std_alpha_w": float(np.std(alpha_w_values)),
        "min_alpha_w": float(np.min(alpha_w_values)),
        "max_alpha_w": float(np.max(alpha_w_values)),
    }
    
    print()
    print(f"  Summary:")
    print(f"    Mean alpha_w = {summary['mean_alpha_w']:.4f} ± {summary['std_alpha_w']:.4f}")
    print(f"    Range: [{summary['min_alpha_w']:.4f}, {summary['max_alpha_w']:.4f}]")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved to: {output_path}")
    
    return {"summary": summary}
