#!/usr/bin/env python3
"""
Simple script to run the Advanced Pipeline with proper configuration.

Usage:
    python run_advanced_pipeline.py --quick          # Quick test (small parameters)
    python run_advanced_pipeline.py                  # Full pipeline
    python run_advanced_pipeline.py --config my_config.json  # Custom config
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add problem directories to path
sys.path.append(str(Path(__file__).parent / "problems"))
sys.path.append(str(Path(__file__).parent / "problems" / "FanZhangWEFNexusModel"))

from pipelines.advanced_pipeline import AdvancedPipelineExperiment

def main():
    """Main function to run the advanced pipeline."""
    parser = argparse.ArgumentParser(description='Run advanced Fan Zhang WEF Nexus pipeline experiment')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced parameters')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output (shows all details)')
    parser.add_argument('--quiet', action='store_true', help='Suppress most output (only show progress)')
    
    args = parser.parse_args()
    
    # Determine verbosity level
    if args.quiet:
        verbosity = False
    elif args.verbose:
        verbosity = True
    else:
        verbosity = False  # Default to quiet mode to reduce noise
    
    # Default configuration
    config = {
        'base_data_file': 'problems/FanZhangWEFNexusModel/data/fan_zhang_wef_nexus_model_problem_data.json',
        'output_dir': 'outputs/experiments/advanced_pipeline',
        'random_seed': 42,
        
        # Scenario generation
        'scenarios_per_profile': 20,  # Reasonable default
        
        # Training data generation
        'max_training_scenarios': 30,  # Use subset for training
        'samples_per_scenario': 10,
        'solve_time_limit': 300,
        
        # Neural network training
        'nn_epochs': 200,
        'nn_learning_rate': 1e-4,
        'nn_batch_size': 32,
        'nn_patience': 20,
        'encoder_hidden_dim': 128,
        'encoder_latent_dim': 64,
        'trunk_hidden': [256, 128, 64],
        
        # Testing
        'max_test_scenarios': 6,
        
        # Verbosity control
        'verbose': verbosity
    }
    
    # Quick test configuration
    if args.quick:
        if verbosity:
            print("🚀 Running QUICK TEST mode")
        config.update({
            'scenarios_per_profile': 2,  # Very small
            'max_training_scenarios': 4,
            'samples_per_scenario': 1,
            'nn_epochs': 5,
            'nn_patience': 2,
            'max_test_scenarios': 2,
            'solve_time_limit': 60,
            'encoder_hidden_dim': 32,
            'encoder_latent_dim': 16,
            'trunk_hidden': [32, 16]
        })
        config['output_dir'] = 'outputs/experiments/quick_test'
    
    # Load custom configuration if provided
    if args.config:
        print(f"📁 Loading custom config from: {args.config}")
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
    
    print("🎯 ADVANCED FAN ZHANG WEF NEXUS PIPELINE")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   Output directory: {config['output_dir']}")
    print(f"   Scenarios per profile: {config['scenarios_per_profile']}")
    print(f"   Training scenarios: {config['max_training_scenarios']}")
    print(f"   Neural network epochs: {config['nn_epochs']}")
    print(f"   Test scenarios: {config['max_test_scenarios']}")
    print("=" * 60)
    
    try:
        # Create and run the experiment
        experiment = AdvancedPipelineExperiment(config)
        
        print("🚀 Starting complete pipeline execution...")
        start_time = time.time()
        
        # Run the complete pipeline
        results = experiment.run_complete_pipeline()
        
        total_time = time.time() - start_time
        
        print("=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Results Summary:")
        print(f"   Total runtime: {total_time:.2f} seconds")
        print(f"   Scenarios generated: {results.get('scenario_count', 'N/A')}")
        print(f"   Model file: {results.get('model_file', 'N/A')}")
        print(f"   Output directory: {results.get('output_directory', 'N/A')}")
        
        # Save results
        experiment.save_results()
        
        print("=" * 60)
        print("✅ All outputs saved successfully!")
        print(f"📁 Check the '{config['output_dir']}' directory for all results")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print("=" * 60)
        print("❌ PIPELINE FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        print("💡 Try running with --quick flag for faster debugging")
        print("💡 Or check the step-by-step test: python test_advanced_step_by_step.py --step 1")
        print("=" * 60)
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("🎯 Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed. Check errors above.")
        sys.exit(1) 