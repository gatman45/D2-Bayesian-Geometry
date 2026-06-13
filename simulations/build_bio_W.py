"""
build_bio_W.py
==============
Build biologically-constrained weight matrices for spiking neural networks.

Following the criticality framework: σ_B ≈ 1.04 (spectral radius at edge of chaos).
"""

import numpy as np
from scipy.linalg import eigvalsh


def build_weight_matrix(N: int = 500, sigma_B: float = 1.04, seed: int = 0) -> np.ndarray:
    """
    Build a biologically-constrained weight matrix with spectral radius σ_B.
    
    Args:
        N: Network size (number of neurons)
        sigma_B: Target spectral radius (default: 1.04, criticality threshold)
        seed: Random seed for reproducibility
    
    Returns:
        W: Weight matrix (N, N) with spectral radius ≈ sigma_B
    """
    np.random.seed(seed)
    
    # Generate random sparse connectivity (10% connectivity)
    connectivity = 0.1
    W_sparse = np.random.randn(N, N) * (np.random.rand(N, N) < connectivity)
    
    # Normalize to target spectral radius
    eigenvalues = eigvalsh(W_sparse @ W_sparse.T)
    max_eigenvalue = np.sqrt(np.max(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    
    if max_eigenvalue > 0:
        W = W_sparse * (sigma_B / max_eigenvalue)
    else:
        W = W_sparse
    
    return W
