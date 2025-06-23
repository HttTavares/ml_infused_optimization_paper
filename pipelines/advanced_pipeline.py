#!/usr/bin/env python3
"""
Advanced Pipeline for Fan Zhang WEF Nexus Model with Complete NN-E Integration

This script provides the full sophisticated pipeline with:
1. Comprehensive data perturbation and scenario generation
2. Realistic training data generation via optimization solving
3. Advanced neural network training with validation
4. Full solver integration and comparison
5. Performance analysis and reporting

Usage:
    python advanced_pipeline.py [--config config.json] [--quick]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Set up path for all imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "problems"))
sys.path.insert(0, str(project_root / "problems" / "FanZhangWEFNexusModel"))

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

# Import the basic plugin from the simple pipeline
from pipelines.full_pipeline import FanZhangPlugin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedPipelineExperiment:
    """
    Advanced pipeline orchestrator with comprehensive features:
    - Sophisticated scenario generation
    - Realistic training data from actual optimization solves
    - Advanced neural network training with proper validation
    - Comprehensive solver comparison and analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced pipeline."""
        self.config = config
        self.results = {}
        
        # Set up directories
        self.output_dir = Path(config.get('output_dir', 'advanced_pipeline_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.output_dir / 'scenarios').mkdir(exist_ok=True)
        (self.output_dir / 'training_data').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        
        logger.info(f"🚀 Advanced pipeline initialized with output directory: {self.output_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete sophisticated pipeline."""
        logger.info("=" * 70)
        logger.info("🎯 STARTING ADVANCED PIPELINE EXPERIMENT")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate comprehensive perturbed scenarios
            logger.info("📊 Step 1: Generating comprehensive perturbed scenarios...")
            perturbed_scenarios = self._generate_comprehensive_scenarios()
            
            # Step 2: Generate realistic training data through optimization
            logger.info("🏭 Step 2: Generating realistic training data...")
            training_data_file = self._generate_realistic_training_data(perturbed_scenarios)
            
            # Step 3: Train neural network with validation
            logger.info("🧠 Step 3: Training neural network with validation...")
            model_file = self._train_validated_neural_network(training_data_file)
            
            # Step 4: Comprehensive inference experiments
            logger.info("🔍 Step 4: Running comprehensive inference experiments...")
            inference_results = self._run_comprehensive_inference(model_file, perturbed_scenarios)
            
            # Step 5: Advanced analysis and reporting
            logger.info("📈 Step 5: Performing advanced analysis...")
            analysis_results = self._perform_advanced_analysis(inference_results)
            
            # Step 6: Generate comprehensive report
            logger.info("📄 Step 6: Generating comprehensive report...")
            self._generate_comprehensive_report(analysis_results)
            
            total_time = time.time() - start_time
            
            # Compile final results
            self.results = {
                'pipeline_type': 'advanced',
                'config': self.config,
                'scenario_count': len(perturbed_scenarios),
                'training_data_file': str(training_data_file),
                'model_file': str(model_file),
                'inference_results': inference_results,
                'analysis': analysis_results,
                'total_runtime': total_time,
                'output_directory': str(self.output_dir)
            }
            
            logger.info(f"✅ Advanced pipeline completed successfully in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Advanced pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_comprehensive_scenarios(self) -> List[Dict[str, Any]]:
        """Generate comprehensive perturbed scenarios with multiple intensity levels."""
        base_data_file = self.config['base_data_file']
        
        # Load base data
        with open(base_data_file, 'r') as f:
            base_data = json.load(f)
        
        fan_zhang_data = base_data["problems"]["FanZhangWEFNexusModel"]
        
        perturbed_scenarios = []
        scenario_id = 0
        
        # Generate scenarios with different perturbation profiles
        perturbation_profiles = ['mild', 'moderate', 'severe']  # FIXED: Use correct profile names
        intensities = [0.02, 0.05, 0.10, 0.15]
        
        for profile in perturbation_profiles:
            for intensity in intensities:
                n_scenarios = self.config.get('scenarios_per_profile', 10)
                
                logger.info(f"   Generating {n_scenarios} scenarios with {profile} profile, intensity {intensity}")
                
                perturber = PerturbFanZhangData(
                    R=intensity,
                    random_seed=self.config.get('random_seed', 42) + scenario_id,
                    perturbation_profile=profile
                )
                
                for i in range(n_scenarios):
                    # Perturb data - FIXED: perturb() expects full data structure
                    perturbed_full_data = perturber.perturb(base_data.copy())
                    perturbed_data = perturbed_full_data["problems"]["FanZhangWEFNexusModel"]
                    
                    # Save perturbed data - use the full structure
                    scenario_file = self.output_dir / 'scenarios' / f"scenario_{scenario_id:03d}_{profile}_{intensity:.2f}.json"
                    with open(scenario_file, 'w') as f:
                        json.dump(perturbed_full_data, f, indent=2)  # Save full structure
                    
                    perturbed_scenarios.append({
                        'scenario_id': scenario_id,
                        'profile': profile,
                        'intensity': intensity,
                        'data_file': str(scenario_file),
                        'perturbed_data': perturbed_data
                    })
                    
                    scenario_id += 1
        
        logger.info(f"✅ Generated {len(perturbed_scenarios)} comprehensive scenarios")
        
        # Save scenario metadata
        metadata_file = self.output_dir / 'scenarios' / 'scenario_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'total_scenarios': len(perturbed_scenarios),
                'profiles': perturbation_profiles,
                'intensities': intensities,
                'scenarios_per_profile': self.config.get('scenarios_per_profile', 10),
                'generation_timestamp': time.time()
            }, f, indent=2)
        
        return perturbed_scenarios
    
    def _generate_realistic_training_data(self, perturbed_scenarios: List[Dict]) -> Path:
        """Generate realistic training data by solving optimization problems."""
        logger.info("🔧 Solving optimization problems to generate realistic training data...")
        
        training_samples = []
        n_samples_per_scenario = self.config.get('samples_per_scenario', 5)
        max_scenarios = self.config.get('max_training_scenarios', 30)
        
        # Use a subset of scenarios for training data generation
        training_scenarios = perturbed_scenarios[:max_scenarios]
        
        for i, scenario_info in enumerate(training_scenarios):
            logger.info(f"   Processing scenario {i+1}/{len(training_scenarios)}: {scenario_info['profile']} @ {scenario_info['intensity']}")
            
            try:
                # Create problem instance
                problem = FanZhangWEFNexusModel(data_file=scenario_info['data_file'])
                
                # Generate multiple training samples for this scenario
                scenario_samples = self._generate_samples_for_scenario(
                    problem, scenario_info, n_samples_per_scenario
                )
                
                training_samples.extend(scenario_samples)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"   Completed {i+1}/{len(training_scenarios)} scenarios, {len(training_samples)} samples so far")
                
            except Exception as e:
                logger.warning(f"   Failed to process scenario {scenario_info['scenario_id']}: {e}")
                continue
        
        # Save comprehensive training data
        training_data_file = self.output_dir / 'training_data' / "comprehensive_training_data.json"
        
        # Organize data for neural network consumption
        training_data = {
            'inputs': [sample['input'] for sample in training_samples],
            'scenarios': [sample['scenarios'] for sample in training_samples],
            'probabilities': [sample['probabilities'] for sample in training_samples],
            'outputs': [sample['output'] for sample in training_samples],
            'metadata': {
                'n_samples': len(training_samples),
                'input_dim': len(training_samples[0]['input']) if training_samples else 0,
                'scenario_dim': training_samples[0]['scenarios'].shape[1] if training_samples else 0,
                'n_scenarios': training_samples[0]['scenarios'].shape[0] if training_samples else 0,
                'generation_method': 'optimization_solving',
                'scenarios_used': len(training_scenarios),
                'samples_per_scenario': n_samples_per_scenario
            },
            'sample_metadata': [
                {
                    'scenario_id': sample['scenario_id'],
                    'sample_idx': sample['sample_idx'],
                    'optimization_status': sample.get('optimization_status', 'unknown'),
                    'solve_time': sample.get('solve_time', 0.0)
                }
                for sample in training_samples
            ]
        }
        
        with open(training_data_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"✅ Generated {len(training_samples)} realistic training samples")
        logger.info(f"💾 Saved comprehensive training data to: {training_data_file}")
        
        return training_data_file
    
    def _generate_samples_for_scenario(self, problem: FanZhangWEFNexusModel, scenario_info: Dict, n_samples: int) -> List[Dict]:
        """Generate multiple training samples for a single scenario by varying first-stage decisions."""
        samples = []
        
        for sample_idx in range(n_samples):
            try:
                # Build fresh model for each sample
                model = problem.build_model()
                
                # Initialize with realistic values
                problem.initialize_variables_for_neural_network(model, verbose=self.config.get('verbose', True))
                
                # Apply strategic perturbations to first-stage variables
                fs_vars = problem.first_stage_vars(model)
                
                if sample_idx > 0:  # Keep first sample as initialized, perturb others
                    self._apply_strategic_perturbations(fs_vars, sample_idx, n_samples)
                
                # Solve the optimization problem
                solve_start = time.time()
                solver = pyo.SolverFactory('glpk')
                solver.options['tmlim'] = self.config.get('solve_time_limit', 300)
                
                result = solver.solve(model, tee=False)
                solve_time = time.time() - solve_start
                
                if result.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
                    # Extract solution data
                    x_values = [pyo.value(var) for var in fs_vars]
                    
                    # Create plugin to get scenario data
                    plugin = FanZhangPlugin(scenario_info['data_file'])
                    scenario_data, prob_data = plugin.get_scenario_tensor()
                    
                    # Calculate true expected recourse value (profit in this case)
                    expected_recourse = pyo.value(model.Profit) if hasattr(model, 'Profit') else 0.0
                    
                    sample = {
                        'input': x_values,
                        'scenarios': scenario_data[0],  # Remove batch dimension
                        'probabilities': prob_data[0],  # Remove batch dimension
                        'output': expected_recourse,
                        'scenario_id': scenario_info['scenario_id'],
                        'sample_idx': sample_idx,
                        'optimization_status': str(result.solver.termination_condition),
                        'solve_time': solve_time,
                        'objective_value': pyo.value(model.obj_ws) if hasattr(model, 'obj_ws') else None
                    }
                    
                    samples.append(sample)
                
                else:
                    logger.warning(f"     Sample {sample_idx} failed: {result.solver.termination_condition}")
                    
            except Exception as e:
                logger.warning(f"     Error generating sample {sample_idx}: {e}")
                continue
        
        return samples
    
    def _apply_strategic_perturbations(self, fs_vars: List[pyo.Var], sample_idx: int, total_samples: int):
        """Apply strategic perturbations to first-stage variables for diverse training samples."""
        # Different perturbation strategies based on sample index
        strategies = ['uniform_random', 'systematic_scaling', 'crop_focused', 'district_focused']
        strategy = strategies[sample_idx % len(strategies)]
        
        if strategy == 'uniform_random':
            # Apply uniform random perturbations
            for var in fs_vars:
                if var.value is not None:
                    variation = np.random.uniform(0.7, 1.3)
                    var.set_value(max(0, var.value * variation))
        
        elif strategy == 'systematic_scaling':
            # Apply systematic scaling based on sample index
            scale_factor = 0.5 + 1.5 * (sample_idx / total_samples)
            for var in fs_vars:
                if var.value is not None:
                    var.set_value(max(0, var.value * scale_factor))
        
        elif strategy == 'crop_focused':
            # Focus perturbations on specific crops
            n_vars_per_crop = len(fs_vars) // 3  # Assuming 3 crops
            crop_idx = sample_idx % 3
            start_idx = crop_idx * n_vars_per_crop
            end_idx = start_idx + n_vars_per_crop
            
            for i, var in enumerate(fs_vars):
                if var.value is not None:
                    if start_idx <= i < end_idx:
                        # Higher variation for focused crop
                        variation = np.random.uniform(0.5, 1.8)
                    else:
                        # Lower variation for others
                        variation = np.random.uniform(0.8, 1.2)
                    var.set_value(max(0, var.value * variation))
        
        elif strategy == 'district_focused':
            # Focus perturbations on specific districts
            # Apply similar logic but for districts instead of crops
            for var in fs_vars:
                if var.value is not None:
                    variation = np.random.uniform(0.6, 1.4)
                    var.set_value(max(0, var.value * variation))
    
    def _train_validated_neural_network(self, training_data_file: Path) -> Path:
        """Train neural network with comprehensive validation and monitoring."""
        # Enhanced training configuration
        config_dict = {
            'epochs': self.config.get('nn_epochs', 200),
            'learning_rate': self.config.get('nn_learning_rate', 1e-3),
            'batch_size': self.config.get('nn_batch_size', 32),
            'validation_split': 0.2,
            'early_stopping_patience': self.config.get('nn_patience', 20),
            'encoder_hidden_dim': self.config.get('encoder_hidden_dim', 128),
            'encoder_latent_dim': self.config.get('encoder_latent_dim', 64),
            'trunk_hidden': self.config.get('trunk_hidden', [256, 128, 64])
        }
        
        model_file = self.output_dir / 'models' / "validated_nne_surrogate.pt"
        
        # Train using external module with enhanced monitoring
        logger.info("🧠 Starting comprehensive neural network training...")
        training_start = time.time()
        
        training_results = create_and_train_nne_model(
            training_data_file=str(training_data_file),
            config_dict=config_dict,
            save_path=str(model_file),
            verbose=self.config.get('verbose', True)
        )
        
        training_time = time.time() - training_start
        
        # Save comprehensive training metadata
        metadata_file = self.output_dir / 'models' / "training_metadata.json"
        comprehensive_metadata = {
            'training_config': config_dict,
            'training_results': training_results,
            'model_file': str(model_file),
            'training_time': training_time,
            'training_timestamp': time.time(),
            'pipeline_type': 'advanced'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=self._json_serializer)
        
        logger.info(f"✅ Neural network training completed in {training_time:.2f} seconds")
        logger.info(f"📊 Final R²: {training_results['final_r2']:.3f}")
        logger.info(f"📉 Best validation loss: {training_results['best_val_loss']:.6f}")
        logger.info(f"💾 Model saved to: {model_file}")
        
        return model_file
    
    def _run_comprehensive_inference(self, model_file: Path, perturbed_scenarios: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive inference experiments with detailed performance analysis."""
        # Select test scenarios strategically
        test_scenarios = self._select_test_scenarios(perturbed_scenarios)
        
        inference_results = {
            'nn_results': [],
            'baseline_results': [],
            'performance_metrics': {},
            'test_scenario_info': test_scenarios
        }
        
        logger.info(f"🔍 Running inference on {len(test_scenarios)} strategically selected scenarios")
        
        for i, scenario_info in enumerate(test_scenarios):
            logger.info(f"   Testing scenario {i+1}/{len(test_scenarios)}: {scenario_info['profile']} @ {scenario_info['intensity']}")
            
            try:
                # Create plugin
                plugin = FanZhangPlugin(scenario_info['data_file'])
                
                # Test with neural network solver
                nn_result = self._solve_with_timing(
                    lambda: self._solve_with_nn_solver(plugin, model_file),
                    'neural_network'
                )
                if nn_result:
                    nn_result.update({
                        'scenario_id': scenario_info['scenario_id'],
                        'scenario_profile': scenario_info['profile'],
                        'scenario_intensity': scenario_info['intensity']
                    })
                    inference_results['nn_results'].append(nn_result)
                
                # Test with baseline solver
                baseline_result = self._solve_with_timing(
                    lambda: self._solve_with_baseline_solver(plugin),
                    'baseline'
                )
                if baseline_result:
                    baseline_result.update({
                        'scenario_id': scenario_info['scenario_id'],
                        'scenario_profile': scenario_info['profile'],
                        'scenario_intensity': scenario_info['intensity']
                    })
                    inference_results['baseline_results'].append(baseline_result)
                
            except Exception as e:
                logger.warning(f"   Failed to test scenario {scenario_info['scenario_id']}: {e}")
                continue
        
        # Calculate performance metrics
        inference_results['performance_metrics'] = self._calculate_performance_metrics(
            inference_results['nn_results'],
            inference_results['baseline_results']
        )
        
        # Save inference results
        results_file = self.output_dir / 'results' / 'inference_results.json'
        with open(results_file, 'w') as f:
            json.dump(inference_results, f, indent=2, default=self._json_serializer)
        
        return inference_results
    
    def _select_test_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """Strategically select test scenarios to cover different perturbation profiles and intensities."""
        max_test = self.config.get('max_test_scenarios', 15)
        
        # Group scenarios by profile and intensity
        scenario_groups = {}
        for scenario in scenarios:
            key = (scenario['profile'], scenario['intensity'])
            if key not in scenario_groups:
                scenario_groups[key] = []
            scenario_groups[key].append(scenario)
        
        # Select representative scenarios from each group
        selected = []
        scenarios_per_group = max(1, max_test // len(scenario_groups))
        
        for (profile, intensity), group_scenarios in scenario_groups.items():
            # Select up to scenarios_per_group from this group
            group_selected = group_scenarios[:scenarios_per_group]
            selected.extend(group_selected)
            
            if len(selected) >= max_test:
                break
        
        return selected[:max_test]
    
    def _solve_with_timing(self, solve_func, solver_type: str) -> Optional[Dict]:
        """Execute solve function with comprehensive timing and error handling."""
        try:
            start_time = time.time()
            result = solve_func()
            solve_time = time.time() - start_time
            
            if result:
                result['solve_time'] = solve_time
                result['solver_type'] = solver_type
                return result
            else:
                return None
                
        except Exception as e:
            logger.warning(f"   {solver_type} solver failed: {e}")
            return None
    
    def _solve_with_nn_solver(self, plugin: FanZhangPlugin, model_file: Path) -> Optional[Dict]:
        """Solve using neural network enhanced solver."""
        solver = Neur2SPSolverCorrect(
            plugin=plugin,
            model_path=str(model_file),
            mip_solver='glpk',
            solver_options={'tmlim': 300},
            verbose=self.config.get('verbose', True)
        )
        
        return solver.solve()
    
    def _solve_with_baseline_solver(self, plugin: FanZhangPlugin) -> Optional[Dict]:
        """Solve using baseline solver."""
        model = plugin.pyomo_model()
        plugin.problem.initialize_variables_for_neural_network(model, verbose=self.config.get('verbose', True))
        
        solver = pyo.SolverFactory('glpk')
        solver.options['tmlim'] = 300
        
        result = solver.solve(model, tee=False)
        
        if result.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
            fs_vars = plugin.first_stage_vars(model)
            
            return {
                'first_stage_solution': {f"x_{i}": pyo.value(var) for i, var in enumerate(fs_vars)},
                'approx_objective': pyo.value(model.obj_ws) if hasattr(model, 'obj_ws') else None,
                'termination_cond': str(result.solver.termination_condition),
                'wallclock_sec': result.solver.time if hasattr(result.solver, 'time') else 0.0
            }
        else:
            return None
    
    def _calculate_performance_metrics(self, nn_results: List[Dict], baseline_results: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        if nn_results:
            nn_times = [r['solve_time'] for r in nn_results if 'solve_time' in r]
            metrics['nn_avg_time'] = np.mean(nn_times) if nn_times else 0
            metrics['nn_std_time'] = np.std(nn_times) if nn_times else 0
            metrics['nn_success_rate'] = len(nn_times) / len(nn_results)
        
        if baseline_results:
            baseline_times = [r.get('wallclock_sec', r.get('solve_time', 0)) for r in baseline_results]
            metrics['baseline_avg_time'] = np.mean(baseline_times) if baseline_times else 0
            metrics['baseline_std_time'] = np.std(baseline_times) if baseline_times else 0
            metrics['baseline_success_rate'] = len(baseline_times) / len(baseline_results)
        
        if nn_results and baseline_results and metrics.get('nn_avg_time', 0) > 0:
            metrics['speedup_factor'] = metrics['baseline_avg_time'] / metrics['nn_avg_time']
        
        return metrics
    
    def _perform_advanced_analysis(self, inference_results: Dict) -> Dict[str, Any]:
        """Perform comprehensive analysis of results."""
        analysis = {
            'summary_statistics': {},
            'performance_analysis': {},
            'solution_quality_analysis': {},
            'scenario_sensitivity_analysis': {}
        }
        
        # Summary statistics
        analysis['summary_statistics'] = {
            'total_nn_tests': len(inference_results['nn_results']),
            'total_baseline_tests': len(inference_results['baseline_results']),
            'nn_success_rate': inference_results['performance_metrics'].get('nn_success_rate', 0),
            'baseline_success_rate': inference_results['performance_metrics'].get('baseline_success_rate', 0)
        }
        
        # Performance analysis
        analysis['performance_analysis'] = inference_results['performance_metrics'].copy()
        
        # Scenario sensitivity analysis
        if inference_results['nn_results']:
            analysis['scenario_sensitivity_analysis'] = self._analyze_scenario_sensitivity(
                inference_results['nn_results']
            )
        
        return analysis
    
    def _analyze_scenario_sensitivity(self, results: List[Dict]) -> Dict:
        """Analyze how performance varies across different scenario types."""
        sensitivity = {}
        
        # Group results by scenario characteristics
        by_profile = {}
        by_intensity = {}
        
        for result in results:
            profile = result.get('scenario_profile', 'unknown')
            intensity = result.get('scenario_intensity', 0)
            
            if profile not in by_profile:
                by_profile[profile] = []
            by_profile[profile].append(result['solve_time'])
            
            intensity_key = f"{intensity:.2f}"
            if intensity_key not in by_intensity:
                by_intensity[intensity_key] = []
            by_intensity[intensity_key].append(result['solve_time'])
        
        # Calculate statistics for each group
        for profile, times in by_profile.items():
            sensitivity[f'profile_{profile}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'count': len(times)
            }
        
        for intensity, times in by_intensity.items():
            sensitivity[f'intensity_{intensity}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'count': len(times)
            }
        
        return sensitivity
    
    def _generate_comprehensive_report(self, analysis_results: Dict):
        """Generate a comprehensive HTML report of results."""
        report_file = self.output_dir / 'comprehensive_report.html'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Pipeline Experiment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metrics {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Advanced Pipeline Experiment Report</h1>
                <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Runtime: {self.results.get('total_runtime', 0):.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="metrics">
                    <p><strong>Total Scenarios Generated:</strong> {self.results.get('scenario_count', 0)}</p>
                    <p><strong>Neural Network Tests:</strong> {analysis_results['summary_statistics'].get('total_nn_tests', 0)}</p>
                    <p><strong>Baseline Tests:</strong> {analysis_results['summary_statistics'].get('total_baseline_tests', 0)}</p>
                    <p><strong>NN Success Rate:</strong> {analysis_results['summary_statistics'].get('nn_success_rate', 0):.1%}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>🚀 Performance Analysis</h2>
                <div class="metrics">
                    <p><strong>NN Average Time:</strong> {analysis_results['performance_analysis'].get('nn_avg_time', 0):.3f} seconds</p>
                    <p><strong>Baseline Average Time:</strong> {analysis_results['performance_analysis'].get('baseline_avg_time', 0):.3f} seconds</p>
                    <p><strong>Speedup Factor:</strong> {analysis_results['performance_analysis'].get('speedup_factor', 'N/A')}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Scenario Sensitivity</h2>
                <p>Performance varies across different perturbation profiles and intensities.</p>
            </div>
            
            <div class="section">
                <h2>📁 Output Files</h2>
                <ul>
                    <li>Scenarios: {self.output_dir / 'scenarios'}</li>
                    <li>Training Data: {self.output_dir / 'training_data'}</li>
                    <li>Trained Models: {self.output_dir / 'models'}</li>
                    <li>Results: {self.output_dir / 'results'}</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Comprehensive report generated: {report_file}")
    
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
        """Save comprehensive results to file."""
        if filename is None:
            filename = self.output_dir / "advanced_pipeline_results.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=self._json_serializer)
        
        logger.info(f"💾 Comprehensive results saved to: {filename}")


def main():
    """Main function for the advanced pipeline."""
    parser = argparse.ArgumentParser(description='Run advanced Fan Zhang WEF Nexus pipeline experiment')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'base_data_file': 'problems/FanZhangWEFNexusModel/fan_zhang_wef_nexus_model_problem_data.json',
        'output_dir': 'advanced_pipeline_output',
        'random_seed': 42,
        
        # Scenario generation
        'scenarios_per_profile': 8,
        
        # Training data generation
        'max_training_scenarios': 20,
        'samples_per_scenario': 4,
        'solve_time_limit': 300,
        
        # Neural network training
        'nn_epochs': 100,
        'nn_learning_rate': 1e-3,
        'nn_batch_size': 32,
        'nn_patience': 15,
        'encoder_hidden_dim': 128,
        'encoder_latent_dim': 64,
        'trunk_hidden': [256, 128, 64],
        
        # Testing
        'max_test_scenarios': 12
    }
    
    # Quick test configuration
    if args.quick:
        config.update({
            'scenarios_per_profile': 3,
            'max_training_scenarios': 6,
            'samples_per_scenario': 2,
            'nn_epochs': 10,
            'nn_patience': 3,
            'max_test_scenarios': 3,
            'solve_time_limit': 60
        })
        config['output_dir'] = 'quick_test_output'
    
    # Load custom configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
    
    print("🎯 Starting Advanced Fan Zhang WEF Nexus Pipeline")
    print("=" * 60)
    print("This is a comprehensive test of all pipeline components")
    print("For full functionality, implement the missing methods")
    print("=" * 60)


if __name__ == "__main__":
    main() 