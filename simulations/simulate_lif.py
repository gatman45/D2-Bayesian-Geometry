"""
simulate_lif.py
===============
Leaky Integrate-and-Fire (LIF) network simulator.
Runs a biologically-constrained SNN and detects neuronal avalanches
following Beggs & Plenz (2003).

Usage:
    python simulate_lif.py
    # or import:
    from simulations.simulate_lif import simulate_lif_network
"""

import numpy as np
from typing import Tuple, List, Dict


def simulate_lif_network(
    W: np.ndarray,
    T: float = 2.0,
    dt: float = 0.001,
    tau_m: float = 0.020,
    V_th: float = 1.0,
    V_reset: float = 0.0,
    tau_ref: float = 0.002,
    input_rate: float = 1.0,
    seed: int = 0,
) -> Dict:
    """
    Simulate a LIF network driven by Poisson input.

    Parameters
    ----------
    W          : weight matrix (N x N)
    T          : simulation duration in seconds
    dt         : time step in seconds
    tau_m      : membrane time constant (s)
    V_th       : firing threshold (normalized)
    V_reset    : reset potential
    tau_ref    : refractory period (s)
    input_rate : external Poisson input rate (Hz per neuron)
    seed       : random seed

    Returns
    -------
    dict with keys:
        spikes      : list of (time_bin, neuron_id) spike events
        V_history   : membrane potential history (T_steps x N)
        avalanches  : list of avalanche sizes
        firing_rate : mean firing rate (Hz)
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    T_steps = int(T / dt)

    V = rng.uniform(0, 0.5, size=N)
    refractory_timer = np.zeros(N)

    spikes = []
    V_history = np.zeros((min(T_steps, 500), N))  # store first 500 steps
    total_spikes = 0

    for t in range(T_steps):
        # External Poisson input
        I_ext = (rng.uniform(size=N) < input_rate * dt).astype(float)

        # Recurrent input from previous spikes
        S = np.zeros(N)
        if t > 0 and len(spikes) > 0:
            recent = [s for s in spikes if s[0] == t - 1]
            if recent:
                fired_neurons = [s[1] for s in recent]
                S[fired_neurons] = 1.0

        I_rec = W @ S

        # Update membrane potential (Euler integration of LIF)
        dV = dt / tau_m * (-V + I_rec + I_ext)
        V = V + dV

        # Refractory neurons cannot fire
        V[refractory_timer > 0] = V_reset
        refractory_timer = np.maximum(0, refractory_timer - dt)

        # Spike detection
        fired = np.where(V >= V_th)[0]
        for nid in fired:
            spikes.append((t, int(nid)))
            V[nid] = V_reset
            refractory_timer[nid] = tau_ref
            total_spikes += 1

        if t < 500:
            V_history[t] = V.copy()

    # Compute firing rate
    firing_rate = total_spikes / (N * T)

    # Detect avalanches
    avalanches = detect_avalanches(spikes, T_steps, N)

    print(f"[simulate_lif] N={N}, T={T}s, dt={dt}s")
    print(f"[simulate_lif] Total spikes: {total_spikes}")
    print(f"[simulate_lif] Mean firing rate: {firing_rate:.3f} Hz")
    print(f"[simulate_lif] Avalanches detected: {len(avalanches)}")
    if avalanches:
        print(f"[simulate_lif] Mean avalanche size: {np.mean(avalanches):.2f}")

    return {
        "spikes": spikes,
        "V_history": V_history,
        "avalanches": avalanches,
        "firing_rate": firing_rate,
        "total_spikes": total_spikes,
        "N": N,
        "T": T,
    }


def detect_avalanches(
    spikes: List[Tuple[int, int]],
    T_steps: int,
    N: int,
    bin_factor: float = 4.0,
) -> List[int]:
    """
    Detect avalanches following Beggs & Plenz (2003).
    An avalanche is a continuous sequence of active time bins
    bounded by silent bins.

    Parameters
    ----------
    spikes     : list of (time_bin, neuron_id)
    T_steps    : total number of time bins
    N          : number of neurons
    bin_factor : avalanche bin = dt * bin_factor

    Returns
    -------
    avalanche_sizes : list of int (number of spikes per avalanche)
    """
    if not spikes:
        return []

    # Count spikes per time bin
    spike_counts = np.zeros(T_steps, dtype=int)
    for t, _ in spikes:
        if t < T_steps:
            spike_counts[t] += 1

    # Detect avalanches as runs of non-zero bins
    avalanches = []
    in_avalanche = False
    current_size = 0

    for count in spike_counts:
        if count > 0:
            in_avalanche = True
            current_size += count
        else:
            if in_avalanche:
                avalanches.append(current_size)
                current_size = 0
                in_avalanche = False

    if in_avalanche and current_size > 0:
        avalanches.append(current_size)

    return avalanches


def compute_branching_parameter(avalanches: List[int]) -> float:
    """
    Estimate the branching parameter m from avalanche size distribution.
    m = 1.0 at criticality (Beggs & Plenz 2003, Priesemann et al. 2014).
    """
    if len(avalanches) < 2:
        return float("nan")
    sizes = np.array(avalanches)
    # Simple estimator: ratio of consecutive generation sizes
    # For a branching process: P(s) ~ s^{-3/2} at criticality
    # Estimator from Priesemann et al. 2014
    log_sizes = np.log(sizes[sizes > 0])
    if len(log_sizes) < 2:
        return float("nan")
    alpha_est = 1 + len(log_sizes) / np.sum(log_sizes - np.min(log_sizes))
    # m = (alpha_est - 2) / (alpha_est - 1) for pure power law
    if alpha_est > 1:
        m = (alpha_est - 2) / (alpha_est - 1)
    else:
        m = float("nan")
    return float(m)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from simulations.build_bio_W import build_weight_matrix

    print("=" * 60)
    print("  SIMULATE_LIF — LIF Network Simulator")
    print("=" * 60)

    W = build_weight_matrix(N=200, seed=0)
    results = simulate_lif_network(W, T=1.0, seed=0)

    avs = results["avalanches"]
    if avs:
        m = compute_branching_parameter(avs)
        print(f"\n[branching] Estimated m = {m:.4f}")
        print(f"  (m ≈ 1.0 expected at criticality, "
              f"observed: {m:.4f})")
    print("=" * 60)
    print("  Simulation complete.")
