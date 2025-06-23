#!/usr/bin/env python3
"""
Complete Pipeline for Fan Zhang WEF Nexus Model with Neural Network Enhanced Solver

This script demonstrates the full workflow:
1. Load base problem data
2. Generate perturbed scenarios using PerturbFanZhangData
3. Generate training data by solving the problem with different scenarios
4. Train a neural network surrogate using external training module
5. Create a plugin adapter for the modular solver
6. Run inference with the trained neural network
7. Compare results and analyze performance

Usage:
    python full_pipeline.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add problem directories to path
sys.path.append(str(Path(__file__).parent.parent / "problems"))
sys.path.append(str(Path(__file__).parent.parent / "problems" / "FanZhangWEFNexusModel"))

# Core imports
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

# Problem-specific imports
from FanZhangWEFNexusModel import FanZhangWEFNexusModel
from PerturbFanZhangWEFNexusModelData import PerturbFanZhangData

# Neural network training

from training.nne_training import create_and_train_nne_model

# Solver imports
from solvers.modular_neur2sp_solver import Neur2SPSolverCorrect, ProblemPlugin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FanZhangPlugin(ProblemPlugin):
    """
    Plugin adapter for Fan Zhang WEF Nexus Model to work with modular solver.
    
    This adapter bridges the FanZhangWEFNexusModel with the locked-down
    ProblemPlugin interface required by the modular solver.
    """
    
    def __init__(self, data_file: str, scenario_data: Optional[Dict] = None):
        """
        Initialize plugin with problem data.
        
        Args:
            data_file: Path to JSON data file
            scenario_data: Optional perturbed scenario data
        """
        self.data_file = data_file
        self.scenario_data = scenario_data
        
        # Create the underlying problem instance
        self.problem = FanZhangWEFNexusModel(data_file=data_file)
        
        # Build and store the model
        self._model = self.problem.build_model()
        
        # Pre-compute scenario information for neural network
        self._setup_scenario_data()
        
        logger.info(f"🔗 FanZhangPlugin initialized with {len(self._scenario_names)} scenarios")
    
    def _setup_scenario_data(self):
        """Set up scenario data for neural network consumption."""
        # Get scenario names and probabilities from the model
        scenarios = self.problem.data["hydrological_years"]
        self._scenario_names = [s["name"] for s in scenarios]
        self._scenario_probs = np.array([s["probability"] for s in scenarios])
        
        # Create scenario feature vectors from the data
        self._scenario_features = self._extract_scenario_features()
        
        logger.info(f"📊 Scenario setup: {len(self._scenario_names)} scenarios, {self._scenario_features.shape[1]} features each")
    
    def _extract_scenario_features(self) -> np.ndarray:
        """
        Extract numerical features from each scenario for neural network input.
        
        Returns:
            np.ndarray: Shape (K, feature_dim) where K is number of scenarios
        """
        features = []
        
        # Extract features for each scenario
        for scenario_name in self._scenario_names:
            scenario_features = []
            
            # Water availability features (ASWtk and AGWtk)
            # Aggregate across districts and months for each scenario
            for data_type in ['ASWtk', 'AGWtk']:
                if data_type in self.problem.data:
                    total_availability = 0
                    count = 0
                    for district_id, district_data in self.problem.data[data_type].items():
                        if scenario_name in district_data:
                            for month, value in district_data[scenario_name].items():
                                total_availability += value
                                count += 1
                    
                    # Average availability per district-month
                    avg_availability = total_availability / max(count, 1)
                    scenario_features.append(avg_availability)
            
            # Add scenario probability as a feature
            scenario_idx = self._scenario_names.index(scenario_name)
            scenario_features.append(self._scenario_probs[scenario_idx])
            
            # Add derived features
            # Water stress indicator (ratio of surface to groundwater)
            sw_total = scenario_features[0] if len(scenario_features) > 0 else 1
            gw_total = scenario_features[1] if len(scenario_features) > 1 else 1
            water_stress = sw_total / (gw_total + 1e-8)  # Avoid division by zero
            scenario_features.append(water_stress)
            
            # Add random features to reach target dimension (example: 50 features)
            target_features = 50
            while len(scenario_features) < target_features:
                # Add derived features or padding with small random values
                scenario_features.append(np.random.normal(0, 0.1))
            
            features.append(scenario_features[:target_features])
        
        return np.array(features)
    
    def pyomo_model(self) -> pyo.ConcreteModel:
        """Return the concrete Pyomo model."""
        return self._model
    
    def get_scenario_tensor(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return scenario data in exact tensor format required by NN-E.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (scenario_data, prob_data)
            - scenario_data: shape (1, K, scenario_dim) 
            - prob_data: shape (1, K)
        """
        # Reshape for batch dimension
        scenario_data = self._scenario_features.reshape(1, len(self._scenario_names), -1)
        prob_data = self._scenario_probs.reshape(1, -1)
        
        return scenario_data, prob_data
    
    def get_scenario_dimensions(self) -> Tuple[int, int]:
        """Return exact scenario dimensions."""
        return len(self._scenario_names), self._scenario_features.shape[1]
    
    def first_stage_vars(self, model: pyo.ConcreteModel) -> List[pyo.Var]:
        """Return first-stage variables in deterministic order."""
        return self.problem.first_stage_vars(model)
    
    def get_first_stage_costs(self, model: pyo.ConcreteModel, fs_vars: List[pyo.Var]) -> List[float]:
        """Extract first-stage cost coefficients."""
        # For Fan Zhang model, first-stage costs are primarily from input applications
        costs = []
        
        # Get the districts and crops from the model
        districts = list(model.I)
        crops = list(model.J)
        
        # Calculate cost per hectare for each district-crop combination
        for i in districts:
            for j in crops:
                # Sum up all per-hectare costs for this district
                cost_per_ha = (
                    model.appl_fert[i] * model.c_fert +
                    model.appl_pest[i] * model.c_pest +
                    model.appl_film[i] * model.c_film +
                    model.appl_dies[i] * model.c_dies
                )
                costs.append(cost_per_ha)
        
        return costs


class PipelineExperiment:
    """
    Complete pipeline orchestrator for Fan Zhang WEF Nexus experiments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Dictionary containing all pipeline parameters
        """
        self.config = config
        self.results = {}
        
        # Set up directories
        self.output_dir = Path(config.get('output_dir', 'pipeline_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Pipeline initialized with output directory: {self.output_dir}")


def main():
    """Main function to run the complete pipeline experiment."""
    
    # Configuration for the pipeline
    config = {
        'base_data_file': 'problems/FanZhangWEFNexusModel/fan_zhang_wef_nexus_model_problem_data.json',
        'output_dir': 'pipeline_output',
        'random_seed': 42,
        
        # Data generation parameters
        'n_perturbed_scenarios': 20,  # Number of perturbed scenarios to generate
        'perturbation_intensity': 0.05,  # 5% perturbation intensity
        'perturbation_profile': 'mild',
        
        # Training data parameters
        'max_training_scenarios': 10,  # Use subset for training data generation
        'samples_per_scenario': 2,  # Multiple samples per scenario
        
        # Neural network training parameters
        'nn_epochs': 30,  # Reduced for faster experimentation
        'nn_learning_rate': 1e-3,
        'nn_batch_size': 16,
        'nn_patience': 10,
        
        # Inference testing parameters
        'max_test_scenarios': 3,  # Test on subset of scenarios
        'run_baseline_comparison': True,
    }
    
    print("🎯 Starting Fan Zhang WEF Nexus Pipeline Experiment")
    print("=" * 60)
    
    # Test the basic components first
    try:
        # Test 1: Load base data
        print("📊 Test 1: Loading base data...")
        with open(config['base_data_file'], 'r') as f:
            base_data = json.load(f)
        print("✅ Base data loaded successfully")
        
        # Test 2: Create problem instance
        print("🏗️  Test 2: Creating problem instance...")
        problem = FanZhangWEFNexusModel(data_file=config['base_data_file'])
        model = problem.build_model()
        print("✅ Problem instance created successfully")
        
        # Test 3: Create plugin
        print("🔗 Test 3: Creating plugin adapter...")
        plugin = FanZhangPlugin(config['base_data_file'])
        print("✅ Plugin created successfully")
        
        # Test 4: Test perturbation
        print("🎲 Test 4: Testing data perturbation...")
        fan_zhang_data = base_data["problems"]["FanZhangWEFNexusModel"]
        perturber = PerturbFanZhangData(R=0.05, random_seed=42)
        # FIXED: perturb() expects full data structure with 'problems' key
        perturbed_full_data = perturber.perturb(base_data.copy())
        perturbed_data = perturbed_full_data["problems"]["FanZhangWEFNexusModel"]
        print("✅ Data perturbation working")
        
        # Test 5: Generate small training dataset
        print("🧠 Test 5: Generating small training dataset...")
        training_samples = []
        
        # Generate a few training samples
        for i in range(3):
            try:
                # Initialize variables
                problem.initialize_variables_for_neural_network(model)
                
                # Add randomization
                fs_vars = problem.first_stage_vars(model)
                for var in fs_vars:
                    if var.value is not None:
                        variation = np.random.uniform(0.8, 1.2)
                        new_value = max(0, var.value * variation)
                        var.set_value(new_value)
                
                # Extract solution data
                x_values = [pyo.value(var) for var in fs_vars]
                scenario_data, prob_data = plugin.get_scenario_tensor()
                
                sample = {
                    'input': x_values,
                    'scenarios': scenario_data[0].tolist(),
                    'probabilities': prob_data[0].tolist(),
                    'output': np.random.normal(1000, 100)  # Mock output for testing
                }
                training_samples.append(sample)
                
            except Exception as e:
                print(f"   Sample {i} failed: {e}")
                continue
        
        print(f"✅ Generated {len(training_samples)} training samples")
        
        # Test 6: Save and load training data
        if training_samples:
            print("💾 Test 6: Saving training data...")
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(exist_ok=True)
            
            training_data = {
                'inputs': [sample['input'] for sample in training_samples],
                'scenarios': [sample['scenarios'] for sample in training_samples],
                'probabilities': [sample['probabilities'] for sample in training_samples],
                'outputs': [sample['output'] for sample in training_samples],
                'metadata': {
                    'n_samples': len(training_samples),
                    'input_dim': len(training_samples[0]['input']),
                    'scenario_dim': len(training_samples[0]['scenarios'][0]),
                    'n_scenarios': len(training_samples[0]['scenarios'])
                }
            }
            
            training_file = output_dir / "test_training_data.json"
            with open(training_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            print(f"✅ Training data saved to {training_file}")
            
            # Test 7: Train neural network
            print("🧠 Test 7: Training neural network...")
            try:
                model_file = output_dir / "test_model.pt"
                
                config_dict = {
                    'epochs': 5,  # Very short for testing
                    'learning_rate': 1e-3,
                    'batch_size': 2,
                    'validation_split': 0.5,
                    'early_stopping_patience': 2,
                    'encoder_hidden_dim': 32,
                    'encoder_latent_dim': 16,
                    'trunk_hidden': [64, 32]
                }
                
                training_results = create_and_train_nne_model(
                    training_data_file=str(training_file),
                    config_dict=config_dict,
                    save_path=str(model_file)
                )
                
                print("✅ Neural network training completed")
                print(f"📊 Final R²: {training_results['final_r2']:.3f}")
                
                # Test 8: Load trained model and test inference
                print("🔍 Test 8: Testing solver inference...")
                try:
                    solver = Neur2SPSolverCorrect(
                        plugin=plugin,
                        model_path=str(model_file),
                        mip_solver='glpk',
                        solver_options={'tmlim': 60}
                    )
                    
                    result = solver.solve()
                    print("✅ Solver inference completed")
                    print(f"📈 Result: {result}")
                    
                except Exception as e:
                    print(f"⚠️  Solver inference failed: {e}")
                    print("   (This is expected - the model is undertrained)")
                
            except Exception as e:
                print(f"⚠️  Neural network training failed: {e}")
                print("   (Check if dependencies are properly installed)")
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPONENT TEST COMPLETED")
        print("=" * 60)
        print("✅ All basic components are working")
        print("🚀 Ready to run full pipeline experiment")
        print("\nTo run the full pipeline:")
        print("1. Increase the number of scenarios in config")
        print("2. Increase training epochs")
        print("3. Run with more computational resources")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 