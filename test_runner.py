#!/usr/bin/env python3
"""
TEST RUNNER - D2 Bayesian Geometry Paper
=========================================
Quick test with --fast mode to verify all modules work correctly.
"""

import subprocess
import sys
import os

def run_test():
    """Run the full D2 paper analysis in fast mode."""
    print("=" * 80)
    print("🚀 STARTING D2-BAYESIAN-GEOMETRY TEST")
    print("=" * 80)
    print()
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the main script with --fast flag
    cmd = [sys.executable, "d2_paper_generator.py", "--fast", "--seeds", "5"]
    print(f"📝 Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_test()
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ TEST PASSED - All modules working correctly!")
    else:
        print("❌ TEST FAILED - Check output above for errors")
    print("=" * 80)
    sys.exit(exit_code)
