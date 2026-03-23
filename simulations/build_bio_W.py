"""
build_bio_W.py
==============
Constructs a biologically-constrained weight matrix W for a spiking
neural network, following:
  - Dale's law (excitatory neurons project only positive weights,
    inhibitory neurons project only negative weights)
  - Braitenberg & Schuez (1998) E/I ratio: 83.2% exc / 16.8% inh
  - Sparse connectivity: connection probability p = 0.10
  - Exact spectral radius control: sigma_B = 1.04

NOTE: The theoretical derivation of sigma_B = 1.04 is part of the
companion paper (in preparation). This script treats sigma_B as a
fixed empirical parameter.

Usage:
    python build_bio_W.py
    # or import as module:
    from simulations.build_bio_W import build_weight_matrix
"""

import numpy as np


def build_weight_matrix(
    N: int = 500,
    f_exc: float = 0.832,
    p: float = 0.10,
    g: float = 4.0,
    sigma_B: float = 1.04,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a biologically-constrained weight matrix.

    Parameters
    ----------
    N       : int   — total number of neurons
    f_exc   : float — fraction of excitatory neurons (Braitenberg & Schuez 1998)
    p       : float — connection probability (sparse cortex)
    g       : float — inhibitory gain factor (van Vreeswijk & Sompolinsky 1996)
    sigma_B : float — target spectral radius
    seed    : int   — random seed for reproducibility

    Returns
    -------
    W : np.ndarray, shape (N, N)
        Weight matrix with spectral radius == sigma_B (up to floating point).
    """
    rng = np.random.default_rng(seed)

    N_exc = int(round(N * f_exc))
    N_inh = N - N_exc

    # --- Step 1: Binary connectivity mask (no autapses) ---
    C = (rng.uniform(size=(N, N)) < p).astype(float)
    np.fill_diagonal(C, 0.0)

    # --- Step 2: Draw raw weights ---
    W = np.zeros((N, N))

    # Excitatory neurons (rows 0..N_exc-1) -> positive weights
    scale_exc = 1.0 / np.sqrt(N)
    W[:N_exc, :] = rng.exponential(scale=scale_exc, size=(N_exc, N))

    # Inhibitory neurons (rows N_exc..N-1) -> negative weights
    scale_inh = g / np.sqrt(N)
    W[N_exc:, :] = -rng.exponential(scale=scale_inh, size=(N_inh, N))

    # Apply connectivity mask
    W *= C

    # --- Step 3: Rescale to target spectral radius ---
    singular_values = np.linalg.svd(W, compute_uv=False)
    sigma_max = singular_values[0]

    if sigma_max > 1e-10:
        W = W * (sigma_B / sigma_max)

    actual_radius = np.linalg.svd(W, compute_uv=False)[0]

    print(f"[build_bio_W] N={N}, N_exc={N_exc}, N_inh={N_inh}")
    print(f"[build_bio_W] seed={seed}, p={p}, g={g}")
    print(f"[build_bio_W] Target sigma_B={sigma_B:.4f}, "
          f"Achieved sigma_B={actual_radius:.6f}")
    print(f"[build_bio_W] W shape: {W.shape}, "
          f"sparsity: {(W == 0).mean():.3f}")

    return W


def verify_dale_law(W: np.ndarray, N_exc: int) -> bool:
    """Verify Dale's law: excitatory rows >= 0, inhibitory rows <= 0."""
    exc_ok = (W[:N_exc, :] >= 0).all()
    inh_ok = (W[N_exc:, :] <= 0).all()
    print(f"[verify_dale] Excitatory weights >= 0: {exc_ok}")
    print(f"[verify_dale] Inhibitory weights <= 0: {inh_ok}")
    return exc_ok and inh_ok


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    print("=" * 60)
    print("  BUILD_BIO_W — Biological Weight Matrix Constructor")
    print("=" * 60)

    results = []
    for seed in range(5):
        W = build_weight_matrix(N=500, seed=seed)
        N_exc = int(round(500 * 0.832))
        dale_ok = verify_dale_law(W, N_exc)
        sv = np.linalg.svd(W, compute_uv=False)
        results.append({
            "seed": seed,
            "spectral_radius": float(sv[0]),
            "dale_law_ok": bool(dale_ok),
            "mean_exc_weight": float(W[:N_exc, :][W[:N_exc, :] > 0].mean()),
            "mean_inh_weight": float(W[N_exc:, :][W[N_exc:, :] < 0].mean()),
        })
        print()

    print("=" * 60)
    print("  SUMMARY (5 seeds)")
    print("=" * 60)
    for r in results:
        status = "OK" if r["dale_law_ok"] else "FAIL"
        print(f"  seed={r['seed']} | sigma_B={r['spectral_radius']:.4f} | "
              f"Dale={status} | "
              f"w_exc={r['mean_exc_weight']:.4f} | "
              f"w_inh={r['mean_inh_weight']:.4f}")
    print("=" * 60)
    print("  All seeds passed." if all(r["dale_law_ok"] for r in results)
          else "  WARNING: Dale's law violation detected.")
