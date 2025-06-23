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
    python full_pipeline_experiment.py
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
sys.path.append(str(Path(__file__).parent / "problems"))
sys.path.append(str(Path(__file__).parent / "problems" / "FanZhangWEFNexusModel"))

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
                    model.appl_fert[i].value * model.c_fert.value +
                    model.appl_pest[i].value * model.c_pest.value +
                    model.appl_film[i].value * model.c_film.value +
                    model.appl_dies[i].value * model.c_dies.value
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
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline from data generation to inference.
        
        Returns:
            Dict containing all results and metadata
        """
        logger.info("=" * 60)
        logger.info("🎯 STARTING COMPLETE PIPELINE EXPERIMENT")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate perturbed scenarios
            logger.info("📊 Step 1: Generating perturbed scenarios...")
            perturbed_scenarios = self._generate_perturbed_scenarios()
            
            # Step 2: Generate training data
            logger.info("🏭 Step 2: Generating training data...")
            training_data_file = self._generate_training_data(perturbed_scenarios)
            
            # Step 3: Train neural network
            logger.info("🧠 Step 3: Training neural network surrogate...")
            model_file = self._train_neural_network(training_data_file)
            
            # Step 4: Run inference experiments
            logger.info("🔍 Step 4: Running inference experiments...")
            inference_results = self._run_inference_experiments(model_file, perturbed_scenarios)
            
            # Step 5: Analyze and compare results
            logger.info("📈 Step 5: Analyzing results...")
            analysis_results = self._analyze_results(inference_results)
            
            total_time = time.time() - start_time
            
            # Compile final results
            self.results = {
                'config': self.config,
                'perturbed_scenarios': len(perturbed_scenarios),
                'training_data_file': str(training_data_file),
                'model_file': str(model_file),
                'inference_results': inference_results,
                'analysis': analysis_results,
                'total_runtime': total_time
            }
            
            logger.info(f"✅ Pipeline completed successfully in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_perturbed_scenarios(self) -> List[Dict[str, Any]]:
        """Generate perturbed versions of the base data."""
        base_data_file = self.config['base_data_file']
        n_scenarios = self.config.get('n_perturbed_scenarios', 100)
        perturbation_intensity = self.config.get('perturbation_intensity', 0.1)
        
        # Load base data
        with open(base_data_file, 'r') as f:
            base_data = json.load(f)
        
        # Extract Fan Zhang data
        if "problems" in base_data and "FanZhangWEFNexusModel" in base_data["problems"]:
            fan_zhang_data = base_data["problems"]["FanZhangWEFNexusModel"]
        else:
            fan_zhang_data = base_data
        
        # Set up perturbation system
        perturber = PerturbFanZhangData(
            R=perturbation_intensity,
            random_seed=self.config.get('random_seed', 42),
            perturbation_profile=self.config.get('perturbation_profile', 'mild')
        )
        
        perturbed_scenarios = []
        
        for i in range(n_scenarios):
            logger.info(f"   Generating scenario {i+1}/{n_scenarios}")
            
            # Perturb data
            perturbed_data = perturber.perturb(fan_zhang_data.copy())
            
            # Save perturbed data
            scenario_file = self.output_dir / f"perturbed_scenario_{i:03d}.json"
            with open(scenario_file, 'w') as f:
                json.dump({
                    "problems": {
                        "FanZhangWEFNexusModel": perturbed_data
                    }
                }, f, indent=2)
            
            perturbed_scenarios.append({
                'scenario_id': i,
                'data_file': str(scenario_file),
                'perturbed_data': perturbed_data
            })
        
        logger.info(f"✅ Generated {len(perturbed_scenarios)} perturbed scenarios")
        return perturbed_scenarios
    
    def _generate_training_data(self, perturbed_scenarios: List[Dict]) -> Path:
        """Generate training data by solving the problem for different scenarios."""
        logger.info("🔧 Solving problems to generate training data...")
        
        training_samples = []
        n_samples_per_scenario = self.config.get('samples_per_scenario', 3)
        
        for scenario_info in perturbed_scenarios[:self.config.get('max_training_scenarios', 20)]:
            logger.info(f"   Processing scenario {scenario_info['scenario_id']}")
            
            try:
                # Create problem instance with perturbed data
                problem = FanZhangWEFNexusModel(data_file=scenario_info['data_file'])
                
                # Generate multiple samples with different first-stage decisions
                for sample_idx in range(n_samples_per_scenario):
                    sample = self._solve_for_training_sample(problem, scenario_info['scenario_id'], sample_idx)
                    if sample:
                        training_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"   Failed to process scenario {scenario_info['scenario_id']}: {e}")
                continue
        
        # Save training data
        training_data_file = self.output_dir / "training_data.json"
        training_data = {
            'inputs': [sample['input'] for sample in training_samples],
            'scenarios': [sample['scenarios'] for sample in training_samples],
            'probabilities': [sample['probabilities'] for sample in training_samples],
            'outputs': [sample['output'] for sample in training_samples],
            'metadata': {
                'n_samples': len(training_samples),
                'input_dim': len(training_samples[0]['input']) if training_samples else 0,
                'scenario_dim': training_samples[0]['scenarios'].shape[1] if training_samples else 0,
                'n_scenarios': training_samples[0]['scenarios'].shape[0] if training_samples else 0
            }
        }
        
        with open(training_data_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"✅ Generated {len(training_samples)} training samples")
        logger.info(f"💾 Saved training data to: {training_data_file}")
        
        return training_data_file
    
    def _solve_for_training_sample(self, problem: FanZhangWEFNexusModel, scenario_id: int, sample_idx: int) -> Optional[Dict]:
        """Solve the problem instance to create a training sample."""
        try:
            # Build model
            model = problem.build_model()
            
            # Initialize variables with realistic values
            problem.initialize_variables_for_neural_network(model)
            
            # Add some randomization to first-stage decisions for variety
            fs_vars = problem.first_stage_vars(model)
            for var in fs_vars:
                if var.value is not None:
                    # Add ±20% random variation
                    variation = np.random.uniform(0.8, 1.2)
                    new_value = max(0, var.value * variation)
                    var.set_value(new_value)
            
            # Solve the problem
            solver = pyo.SolverFactory('glpk')
            solver.options['tmlim'] = 300  # 5 minute limit
            result = solver.solve(model, tee=False)
            
            if result.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
                # Extract solution
                x_values = [pyo.value(var) for var in fs_vars]
                
                # Create plugin to get scenario data
                plugin = FanZhangPlugin(problem.data_file)
                scenario_data, prob_data = plugin.get_scenario_tensor()
                
                # Calculate expected recourse (simplified - use profit as proxy)
                expected_recourse = pyo.value(model.Profit) if hasattr(model, 'Profit') else 0.0
                
                return {
                    'input': x_values,
                    'scenarios': scenario_data[0],  # Remove batch dimension
                    'probabilities': prob_data[0],  # Remove batch dimension
                    'output': expected_recourse,
                    'scenario_id': scenario_id,
                    'sample_idx': sample_idx
                }
            else:
                logger.warning(f"   Solver failed for scenario {scenario_id}, sample {sample_idx}")
                return None
                
        except Exception as e:
            logger.warning(f"   Error solving scenario {scenario_id}, sample {sample_idx}: {e}")
            return None
    
    def _train_neural_network(self, training_data_file: Path) -> Path:
        """Train neural network using external training module."""
        config_dict = {
            'epochs': self.config.get('nn_epochs', 100),
            'learning_rate': self.config.get('nn_learning_rate', 1e-3),
            'batch_size': self.config.get('nn_batch_size', 16),
            'validation_split': 0.2,
            'early_stopping_patience': self.config.get('nn_patience', 10),
            'encoder_hidden_dim': 128,
            'encoder_latent_dim': 64,
            'trunk_hidden': [256, 128, 64]
        }
        
        model_file = self.output_dir / "trained_nne_surrogate.pt"
        
        # Train using external module
        training_results = create_and_train_nne_model(
            training_data_file=str(training_data_file),
            config_dict=config_dict,
            save_path=str(model_file)
        )
        
        # Save training metadata
        metadata_file = self.output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'training_config': config_dict,
                'training_results': training_results,
                'model_file': str(model_file)
            }, f, indent=2, default=self._json_serializer)
        
        logger.info(f"✅ Neural network training completed")
        logger.info(f"📊 Final R²: {training_results['final_r2']:.3f}")
        logger.info(f"💾 Model saved to: {model_file}")
        
        return model_file
    
    def _run_inference_experiments(self, model_file: Path, perturbed_scenarios: List[Dict]) -> Dict[str, Any]:
        """Run inference experiments using the trained neural network."""
        # Test on a subset of scenarios
        test_scenarios = perturbed_scenarios[:self.config.get('max_test_scenarios', 5)]
        
        inference_results = {
            'nn_results': [],
            'baseline_results': [],
            'comparison_metrics': {}
        }
        
        for scenario_info in test_scenarios:
            logger.info(f"   Testing scenario {scenario_info['scenario_id']}")
            
            try:
                # Create plugin
                plugin = FanZhangPlugin(scenario_info['data_file'])
                
                # Test with neural network solver
                nn_result = self._solve_with_nn_solver(plugin, model_file)
                if nn_result:
                    nn_result['scenario_id'] = scenario_info['scenario_id']
                    inference_results['nn_results'].append(nn_result)
                
                # Test with baseline solver (if requested)
                if self.config.get('run_baseline_comparison', True):
                    baseline_result = self._solve_with_baseline_solver(plugin)
                    if baseline_result:
                        baseline_result['scenario_id'] = scenario_info['scenario_id']
                        inference_results['baseline_results'].append(baseline_result)
                
            except Exception as e:
                logger.warning(f"   Failed to test scenario {scenario_info['scenario_id']}: {e}")
                continue
        
        return inference_results
    
    def _solve_with_nn_solver(self, plugin: FanZhangPlugin, model_file: Path) -> Optional[Dict]:
        """Solve using the neural network enhanced solver."""
        try:
            # Create NN-E solver
            solver = Neur2SPSolverCorrect(
                plugin=plugin,
                model_path=str(model_file),
                mip_solver='glpk',
                solver_options={'tmlim': 300}
            )
            
            # Solve
            start_time = time.time()
            result = solver.solve()
            solve_time = time.time() - start_time
            
            result['solver_type'] = 'neural_network'
            result['solve_time'] = solve_time
            
            return result
            
        except Exception as e:
            logger.warning(f"   NN solver failed: {e}")
            return None
    
    def _solve_with_baseline_solver(self, plugin: FanZhangPlugin) -> Optional[Dict]:
        """Solve using baseline solver for comparison."""
        try:
            model = plugin.pyomo_model()
            
            # Initialize variables
            plugin.problem.initialize_variables_for_neural_network(model)
            
            # Solve with standard solver
            solver = pyo.SolverFactory('glpk')
            solver.options['tmlim'] = 300
            
            start_time = time.time()
            result = solver.solve(model, tee=False)
            solve_time = time.time() - start_time
            
            if result.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
                fs_vars = plugin.first_stage_vars(model)
                
                return {
                    'first_stage_solution': {f"x_{i}": pyo.value(var) for i, var in enumerate(fs_vars)},
                    'approx_objective': pyo.value(model.obj_ws) if hasattr(model, 'obj_ws') else None,
                    'termination_cond': str(result.solver.termination_condition),
                    'wallclock_sec': solve_time,
                    'solver_type': 'baseline'
                }
            else:
                return None
                
        except Exception as e:
            logger.warning(f"   Baseline solver failed: {e}")
            return None
    
    def _analyze_results(self, inference_results: Dict) -> Dict[str, Any]:
        """Analyze and compare inference results."""
        analysis = {
            'summary': {},
            'performance_comparison': {},
            'solution_quality': {}
        }
        
        nn_results = inference_results['nn_results']
        baseline_results = inference_results['baseline_results']
        
        # Summary statistics
        analysis['summary'] = {
            'nn_tests_completed': len(nn_results),
            'baseline_tests_completed': len(baseline_results),
            'nn_success_rate': len(nn_results) / max(1, len(nn_results)),
            'baseline_success_rate': len(baseline_results) / max(1, len(baseline_results))
        }
        
        # Performance comparison
        if nn_results and baseline_results:
            nn_times = [r['solve_time'] for r in nn_results if 'solve_time' in r]
            baseline_times = [r['wallclock_sec'] for r in baseline_results if 'wallclock_sec' in r]
            
            if nn_times and baseline_times:
                analysis['performance_comparison'] = {
                    'nn_avg_time': np.mean(nn_times),
                    'baseline_avg_time': np.mean(baseline_times),
                    'speedup_factor': np.mean(baseline_times) / np.mean(nn_times) if np.mean(nn_times) > 0 else float('inf')
                }
        
        # Solution quality comparison
        # (Could be expanded based on specific metrics of interest)
        analysis['solution_quality'] = {
            'note': 'Solution quality comparison requires domain-specific metrics'
        }
        
        return analysis
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)
    
    def save_results(self, filename: Optional[str] = None):
        """Save pipeline results to file."""
        if filename is None:
            filename = self.output_dir / "pipeline_results.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=self._json_serializer)
        
        logger.info(f"💾 Results saved to: {filename}")

def main():
    """Main function to run the complete pipeline experiment."""
    
    # Configuration for the pipeline
    config = {
        'base_data_file': 'problems/FanZhangWEFNexusModel/fan_zhang_wef_nexus_model_problem_data.json',
        'output_dir': 'pipeline_output',
        'random_seed': 42,
        
        # Data generation parameters
        'n_perturbed_scenarios': 50,  # Number of perturbed scenarios to generate
        'perturbation_intensity': 0.05,  # 5% perturbation intensity
        'perturbation_profile': 'mild',
        
        # Training data parameters
        'max_training_scenarios': 15,  # Use subset for training data generation
        'samples_per_scenario': 2,  # Multiple samples per scenario
        
        # Neural network training parameters
        'nn_epochs': 50,  # Reduced for faster experimentation
        'nn_learning_rate': 1e-3,
        'nn_batch_size': 16,
        'nn_patience': 10,
        
        # Inference testing parameters
        'max_test_scenarios': 5,  # Test on subset of scenarios
        'run_baseline_comparison': True,
    }
    
    # Create and run pipeline
    pipeline = PipelineExperiment(config)
    
    try:
        results = pipeline.run_complete_pipeline()
        
        # Save results
        pipeline.save_results()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 PIPELINE EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"✅ Total scenarios generated: {results['perturbed_scenarios']}")
        print(f"🧠 Neural network training completed")
        print(f"🔍 Inference tests: {len(results['inference_results']['nn_results'])} NN, {len(results['inference_results']['baseline_results'])} baseline")
        print(f"⏱️  Total runtime: {results['total_runtime']:.2f} seconds")
        
        if 'performance_comparison' in results['analysis']:
            perf = results['analysis']['performance_comparison']
            print(f"🚀 Average speedup: {perf.get('speedup_factor', 'N/A'):.2f}x")
        
        print(f"📁 Results saved in: {pipeline.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 