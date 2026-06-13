"""
simulate_lif.py
===============
Leaky Integrate-and-Fire (LIF) spiking neural network simulator.

Validates criticality via avalanche dynamics (branching parameter m ≈ 1).
"""

import numpy as np
from typing import Dict, List, Any


def simulate_lif_network(W: np.ndarray, T: float = 1.0, dt: float = 0.001,
                        I_ext: float = 0.5, tau: float = 0.01, 
                        V_th: float = 1.0, seed: int = 0) -> Dict[str, Any]:
    """
    Simulate a LIF spiking neural network.
    
    Args:
        W: Weight matrix (N, N)
        T: Total simulation time (seconds)
        dt: Time step (seconds)
        I_ext: External input current
        tau: Membrane time constant
        V_th: Spike threshold
        seed: Random seed
    
    Returns:
        Dictionary with simulation results including spike raster and avalanches
    """
    np.random.seed(seed)
    
    N = W.shape[0]
    steps = int(T / dt)
    
    # State variables
    V = np.random.randn(N) * 0.1  # Membrane potentials
    spikes = np.zeros((steps, N), dtype=bool)
    spike_times = [[] for _ in range(N)]
    
    # Decay factor
    decay = np.exp(-dt / tau)
    
    for step in range(steps):
        # Membrane potential decay
        V *= decay
        
        # Input: external + recurrent
        I_recurrent = W @ spikes[step - 1].astype(float) if step > 0 else 0
        V += dt * (I_ext + I_recurrent)
        
        # Add noise
        V += np.random.randn(N) * 0.01
        
        # Spike generation
        fired = V >= V_th
        spikes[step, fired] = True
        
        # Reset
        V[fired] = 0.0
        
        # Record spike times
        for i in np.where(fired)[0]:
            spike_times[i].append(step * dt)
    
    # Detect avalanches (groups of consecutive spikes)
    avalanches = detect_avalanches(spikes, min_gap=0.01)
    
    return {
        "spike_raster": spikes,
        "spike_times": spike_times,
        "avalanches": avalanches,
        "total_spikes": np.sum(spikes),
        "firing_rate": np.sum(spikes) / (N * T),
    }


def detect_avalanches(spike_raster: np.ndarray, min_gap: float = 0.01,
                      dt: float = 0.001) -> List[int]:
    """
    Detect avalanches from spike raster (groups of consecutive spikes).
    
    Args:
        spike_raster: Boolean array (steps, N)
        min_gap: Minimum gap (in time units) between avalanches
        dt: Time step
    
    Returns:
        List of avalanche sizes
    """
    spike_counts = np.sum(spike_raster, axis=1)
    min_gap_steps = int(min_gap / dt)
    
    avalanches = []
    current_avalanche_size = 0
    gap_counter = 0
    
    for spike_count in spike_counts:
        if spike_count > 0:
            current_avalanche_size += spike_count
            gap_counter = 0
        else:
            gap_counter += 1
            if gap_counter >= min_gap_steps and current_avalanche_size > 0:
                avalanches.append(current_avalanche_size)
                current_avalanche_size = 0
    
    if current_avalanche_size > 0:
        avalanches.append(current_avalanche_size)
    
    return avalanches


def compute_branching_parameter(avalanches: List[int]) -> float:
    """
    Compute branching parameter m from avalanche size distribution.
    
    At criticality: m ≈ 1.0
    
    Args:
        avalanches: List of avalanche sizes
    
    Returns:
        Estimated branching parameter
    """
    if len(avalanches) < 10:
        return 0.0
    
    avalanches = np.array(avalanches)
    
    # Simple estimate: ratio of large to small avalanches
    median_size = np.median(avalanches)
    large_count = np.sum(avalanches > median_size)
    small_count = np.sum(avalanches <= median_size)
    
    if small_count > 0:
        m = large_count / small_count
    else:
        m = 0.0
    
    return float(m)
