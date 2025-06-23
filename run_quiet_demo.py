#!/usr/bin/env python3
"""
Demo script showing how to run the advanced pipeline in different verbosity modes.

Usage:
    python run_quiet_demo.py --quiet       # Minimal output, just progress
    python run_quiet_demo.py --verbose     # Detailed output (all prints)
    python run_quiet_demo.py               # Default (quiet mode)
"""

import sys
import time
from pathlib import Path

# Add problem directories to path
sys.path.append(str(Path(__file__).parent / "problems"))
sys.path.append(str(Path(__file__).parent / "problems" / "FanZhangWEFNexusModel"))

def run_quiet_demo():
    """Run a quick demo showing the difference between quiet and verbose modes."""
    print("🎯 ADVANCED PIPELINE VERBOSITY DEMO")
    print("=" * 60)
    print("This demo shows how to control verbosity in the advanced pipeline:")
    print()
    print("📱 QUIET MODE (default):")
    print("   - Only shows main progress steps")
    print("   - Hides area variable initialization details")
    print("   - Hides neural network training epochs")
    print("   - Hides solver technical details")
    print()
    print("📢 VERBOSE MODE:")
    print("   - Shows all detailed output")
    print("   - Useful for debugging and research")
    print()
    print("Command examples:")
    print("   python run_advanced_pipeline.py --quiet")
    print("   python run_advanced_pipeline.py --verbose")
    print("   python run_advanced_pipeline.py           # defaults to quiet")
    print()
    print("Configuration options:")
    print("   --quick     : Fast test with small parameters")
    print("   --verbose   : Show all detailed output")
    print("   --quiet     : Show only progress (default)")
    print("   --config    : Use custom configuration file")
    print()
    
    # Import here after path setup
    from advanced_pipeline import AdvancedPipelineExperiment
    
    # Quick test config
    config_quiet = {
        'base_data_file': 'problems/FanZhangWEFNexusModel/fan_zhang_wef_nexus_model_problem_data.json',
        'output_dir': 'demo_quiet_output',
        'random_seed': 42,
        'scenarios_per_profile': 1,
        'max_training_scenarios': 2,
        'samples_per_scenario': 1,
        'solve_time_limit': 30,
        'nn_epochs': 3,
        'nn_learning_rate': 1e-3,
        'nn_batch_size': 16,
        'nn_patience': 2,
        'encoder_hidden_dim': 32,
        'encoder_latent_dim': 16,
        'trunk_hidden': [32, 16],
        'max_test_scenarios': 1,
        'verbose': False  # QUIET MODE
    }
    
    print("🚀 Running MINI DEMO in QUIET MODE:")
    print("   (Only 1 scenario per profile, 2 training scenarios, 3 epochs)")
    print()
    
    start_time = time.time()
    
    try:
        experiment = AdvancedPipelineExperiment(config_quiet)
        results = experiment.run_complete_pipeline()
        
        total_time = time.time() - start_time
        
        print()
        print("✅ QUIET MODE DEMO COMPLETED!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Scenarios: {results.get('scenario_count', 'N/A')}")
        print(f"   Output: {results.get('output_directory', 'N/A')}")
        print()
        print("🔍 To run with full verbosity, try:")
        print("   python run_advanced_pipeline.py --verbose --quick")
        print()
        print("🎯 For your current high-parameter run, use:")
        print("   python run_advanced_pipeline.py --quiet")
        print("   (This will use your edited parameters with minimal output)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_quiet_demo() 