#!/usr/bin/env python3
"""
REAL TEST RUNNER - D2 Bayesian Geometry Paper
==============================================
Execute actual tests and report failures with detailed diagnostics.
"""

import sys
import os
import traceback
import json

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test 1: All modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    try:
        print("  Importing simulations.build_bio_W...")
        from simulations.build_bio_W import build_weight_matrix
        print("    ✅ OK")
        
        print("  Importing simulations.measure_alpha_w...")
        from simulations.measure_alpha_w import measure_alpha_w, run_full_analysis
        print("    ✅ OK")
        
        print("  Importing simulations.simulate_lif...")
        from simulations.simulate_lif import simulate_lif_network, compute_branching_parameter
        print("    ✅ OK")
        
        print("  Importing d2_paper_generator...")
        import d2_paper_generator
        print("    ✅ OK")
        
        return True, None
    except Exception as e:
        print(f"    ❌ FAILED")
        return False, traceback.format_exc()

def test_build_weight_matrix():
    """Test 2: Weight matrix construction."""
    print("\n" + "="*80)
    print("TEST 2: Weight Matrix Construction")
    print("="*80)
    
    try:
        from simulations.build_bio_W import build_weight_matrix
        import numpy as np
        from scipy.linalg import eigvalsh
        
        print("  Building weight matrix (N=100, sigma_B=1.04, seed=0)...")
        W = build_weight_matrix(N=100, sigma_B=1.04, seed=0)
        print(f"    Shape: {W.shape}")
        print(f"    Type: {W.dtype}")
        print(f"    Non-zero elements: {np.count_nonzero(W)}")
        
        # Check spectral radius
        eigenvalues = eigvalsh(W @ W.T)
        spectral_radius = np.sqrt(np.max(eigenvalues))
        print(f"    Spectral radius: {spectral_radius:.4f} (expected ≈ 1.04)")
        
        assert W.shape == (100, 100), f"Wrong shape: {W.shape}"
        assert W.dtype in [np.float64, np.float32], f"Wrong dtype: {W.dtype}"
        assert 0.9 < spectral_radius < 1.2, f"Spectral radius out of range: {spectral_radius}"
        
        print("    ✅ OK")
        return True, None
    except Exception as e:
        print(f"    ❌ FAILED")
        return False, traceback.format_exc()

def test_measure_alpha_w():
    """Test 3: Alpha-w measurement."""
    print("\n" + "="*80)
    print("TEST 3: Alpha-w Measurement")
    print("="*80)
    
    try:
        from simulations.build_bio_W import build_weight_matrix
        from simulations.measure_alpha_w import measure_alpha_w
        import numpy as np
        
        print("  Building weight matrix...")
        W = build_weight_matrix(N=100, sigma_B=1.04, seed=0)
        
        print("  Measuring alpha_w...")
        alpha_w = measure_alpha_w(W)
        print(f"    alpha_w = {alpha_w:.4f}")
        
        assert isinstance(alpha_w, float), f"Wrong type: {type(alpha_w)}"
        assert 1.0 <= alpha_w <= 5.0, f"alpha_w out of reasonable range: {alpha_w}"
        
        print("    ✅ OK")
        return True, None
    except Exception as e:
        print(f"    ❌ FAILED")
        return False, traceback.format_exc()

def test_lif_simulation():
    """Test 4: LIF simulation."""
    print("\n" + "="*80)
    print("TEST 4: LIF Network Simulation")
    print("="*80)
    
    try:
        from simulations.build_bio_W import build_weight_matrix
        from simulations.simulate_lif import simulate_lif_network, compute_branching_parameter
        
        print("  Building weight matrix (N=50 for speed)...")
        W = build_weight_matrix(N=50, sigma_B=1.04, seed=0)
        
        print("  Running LIF simulation (T=0.1s)...")
        results = simulate_lif_network(W, T=0.1, seed=0)
        
        print(f"    Total spikes: {results['total_spikes']}")
        print(f"    Firing rate: {results['firing_rate']:.2f} Hz")
        print(f"    Avalanches detected: {len(results['avalanches'])}")
        
        if len(results['avalanches']) > 0:
            m = compute_branching_parameter(results['avalanches'])
            print(f"    Branching parameter m: {m:.4f}")
        
        assert isinstance(results, dict), "Results should be a dict"
        assert 'total_spikes' in results, "Missing 'total_spikes'"
        assert 'avalanches' in results, "Missing 'avalanches'"
        
        print("    ✅ OK")
        return True, None
    except Exception as e:
        print(f"    ❌ FAILED")
        return False, traceback.format_exc()

def test_full_analysis():
    """Test 5: Full spectral analysis pipeline."""
    print("\n" + "="*80)
    print("TEST 5: Full Spectral Analysis (3 seeds)")
    print("="*80)
    
    try:
        from simulations.measure_alpha_w import run_full_analysis
        import os
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        print("  Running analysis with 3 seeds, N=50...")
        output = run_full_analysis(N=50, n_seeds=3, sigma_B=1.04,
                                  output_path="results/test_alpha_w.json")
        
        summary = output["summary"]
        print(f"    Mean alpha_w: {summary['mean_alpha_w']:.4f} ± {summary['std_alpha_w']:.4f}")
        print(f"    Range: [{summary['min_alpha_w']:.4f}, {summary['max_alpha_w']:.4f}]")
        
        assert isinstance(summary['mean_alpha_w'], float), "mean_alpha_w should be float"
        assert summary['mean_alpha_w'] > 0, "mean_alpha_w should be positive"
        
        # Check if file was saved
        assert os.path.exists("results/test_alpha_w.json"), "Results file not saved"
        
        print("    ✅ OK")
        return True, None
    except Exception as e:
        print(f"    ❌ FAILED")
        return False, traceback.format_exc()

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " D2 BAYESIAN GEOMETRY - REAL TEST SUITE ".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("Weight Matrix", test_build_weight_matrix),
        ("Alpha-w Measurement", test_measure_alpha_w),
        ("LIF Simulation", test_lif_simulation),
        ("Full Analysis", test_full_analysis),
    ]
    
    results = []
    failed = False
    
    for name, test_func in tests:
        success, error = test_func()
        results.append((name, success, error))
        
        if not success:
            failed = True
            print(f"\n{'='*80}")
            print(f"ERROR DETAILS FOR: {name}")
            print(f"{'='*80}")
            print(error)
            print(f"{'='*80}\n")
            break  # Stop on first failure
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success, _ in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name:<30} {status}")
    
    print("="*80)
    
    if failed:
        print("\n❌ TESTS FAILED - See error details above")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
