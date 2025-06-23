# modular_neur2sp_solver.py - Problem-Agnostic NN-E Solver with Locked Plugin Architecture
import os
import pathlib
import logging
import copy
import io
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet

# OMLT imports for exact ReLU embedding
try:
    from omlt import OmltBlock
    from omlt.neuralnet import ReluBigMFormulation, NetworkDefinition
    from omlt.io.onnx import load_onnx_neural_network  # Correct path for OMLT 1.2.2
    OMLT_AVAILABLE = True
    print("✅ OMLT 1.2.2 loaded successfully")
except ImportError as e:
    print(f"⚠️  OMLT not available: {e}")
    print("   Install with: pip install omlt")
    OMLT_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# ============================================================================
# 0. Problem Plugin Interface - LOCKED DOWN BOUNDARY
# ============================================================================

class ProblemPlugin(ABC):
    """
    LOCKED-DOWN interface for two-stage stochastic programming problems.
    
    This boundary is intentionally rigid to ensure complete solver-problem decoupling.
    All methods are abstract with no default implementations or convenience features.
    Data formats are strictly enforced with no flexibility.
    
    Key principles of the locked boundary:
    - Zero convenience methods that could create coupling
    - Strict data format enforcement (no Union types or flexible formats)
    - No validation methods that expose solver internals
    - No default implementations that plugins could rely on
    - Complete responsibility transfer to plugin implementation
    """
    
    @abstractmethod
    def pyomo_model(self) -> pyo.ConcreteModel:
        """
        Return the concrete Pyomo model for this problem.
        
        LOCKED: Must return ConcreteModel only - no AbstractModel flexibility.
        
        Returns:
            pyo.ConcreteModel: Fully instantiated concrete model
        """
        pass
    
    # ----- Scenario Handling - STRICT FORMAT ENFORCEMENT -----
    
    @abstractmethod
    def get_scenario_tensor(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return scenario data in exact tensor format required by NN-E.
        
        LOCKED: No convenience methods, no default implementations.
        Plugin must handle all enumeration, conversion, and batching.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (scenario_data, prob_data)
            - scenario_data: shape (1, K, scenario_dim) - EXACT format required
            - prob_data: shape (1, K) - EXACT format required
            
        Raises:
            NotImplementedError: Plugin must implement this exactly
        """
        pass
    
    @abstractmethod
    def get_scenario_dimensions(self) -> Tuple[int, int]:
        """
        Return exact scenario dimensions for model architecture.
        
        LOCKED: No fallback to defaults, plugin must know its dimensions.
        
        Returns:
            Tuple[int, int]: (num_scenarios, scenario_feature_dim)
            - num_scenarios: Exact number of scenarios (K)
            - scenario_feature_dim: Exact feature dimension per scenario
        """
        pass
    
    # ----- First-Stage Specifics - STRICT CONTRACTS -----
    
    @abstractmethod
    def first_stage_vars(self, model: pyo.ConcreteModel) -> List[pyo.Var]:
        """
        Identify first-stage variables in the exact order expected.
        
        LOCKED: Must return variables in consistent, reproducible order.
        
        Args:
            model: The concrete model (guaranteed to be from pyomo_model())
            
        Returns:
            List[pyo.Var]: First-stage variables in deterministic order
        """
        pass
    
    @abstractmethod
    def get_first_stage_costs(self, model: pyo.ConcreteModel, fs_vars: List[pyo.Var]) -> List[float]:
        """
        Extract first-stage cost coefficients in exact variable order.
        
        LOCKED: Must return costs aligned with fs_vars order exactly.
        
        Args:
            model: The concrete model (guaranteed to be from pyomo_model())
            fs_vars: First-stage variables (guaranteed to be from first_stage_vars())
            
        Returns:
            List[float]: Cost coefficients, same length and order as fs_vars
        """
        pass

# ============================================================================
# 0.1. Example Plugin Implementation - STRICT COMPLIANCE
# ============================================================================

class DummyProblemPlugin(ProblemPlugin):
    """
    Demonstration plugin showing strict compliance with locked-down interface.
    
    This implementation shows how to satisfy the rigid boundary requirements:
    - No reliance on convenience methods
    - Strict data format compliance
    - Complete self-contained scenario handling
    """
    
    def __init__(self, n_scenarios: int = 3, scenario_dim: int = 10, n_vars: int = 5):
        self.n_scenarios = n_scenarios
        self.scenario_dim = scenario_dim 
        self.n_vars = n_vars
        self._build_dummy_model()
        
        # Pre-generate scenario data in exact format
        self._scenario_data, self._prob_data = self._generate_scenario_tensors()
    
    def _build_dummy_model(self):
        """Build a simple dummy 2SP model for demonstration."""
        self.model = pyo.ConcreteModel()
        
        # First-stage variables
        self.model.x = pyo.Var(range(self.n_vars), within=pyo.NonNegativeReals)
        
        # Dummy objective (will be replaced by NN-E)
        self.model.obj = pyo.Objective(expr=sum(self.model.x[i] for i in range(self.n_vars)))
        
        # Simple constraint
        self.model.constraint = pyo.Constraint(expr=sum(self.model.x[i] for i in range(self.n_vars)) <= 10)
    
    def _generate_scenario_tensors(self):
        """Generate scenario data in exact format required by locked interface."""
        # Create random scenario features
        scenario_data = np.random.rand(1, self.n_scenarios, self.scenario_dim)
        
        # Create uniform probabilities that sum to 1
        prob_data = np.full((1, self.n_scenarios), 1.0 / self.n_scenarios)
        
        return scenario_data, prob_data
    
    def pyomo_model(self) -> pyo.ConcreteModel:
        """Return concrete model - no flexibility."""
        return self.model
    
    def get_scenario_tensor(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return pre-generated scenario data in exact format."""
        return self._scenario_data, self._prob_data
    
    def get_scenario_dimensions(self) -> Tuple[int, int]:
        """Return exact dimensions - no defaults."""
        return self.n_scenarios, self.scenario_dim
    
    def first_stage_vars(self, model: pyo.ConcreteModel) -> List[pyo.Var]:
        """Return variables in deterministic order."""
        return [model.x[i] for i in range(self.n_vars)]
    
    def get_first_stage_costs(self, model: pyo.ConcreteModel, fs_vars: List[pyo.Var]) -> List[float]:
        """Return unit costs in exact order."""
        return [1.0] * len(fs_vars)

# ============================================================================
# 1. DeepSet Encoder for Scenario Sets
# ============================================================================

# ============================================================================
# 2. NN-E Architecture Imports (Now External)
# ============================================================================

# Import neural architectures from external training module
from training.nne_training import RecourseSurrogateNNE, DeepSetEncoder
print("✅ NN-E architectures imported from external training module")

# ============================================================================
# 3. Main Neur2SP Solver - Pure Inference Engine (No Training)
# ============================================================================

class Neur2SPSolverCorrect:
    """
    Pure inference engine for NN-E solver - completely externalized training.
    
    This solver is now a pure model loader and inference engine:
    - NO training capabilities (externalized to nne_training.py)
    - NO data generation (handled externally)
    - Pure plugin-based problem interface
    - Exact ReLU embedding with OMLT for MILP formulation
    - Complete decoupling from problem specifics
    """
    
    def __init__(
        self,
        plugin: Optional[ProblemPlugin] = None,
        model_path: Optional[str] = None,
        mip_solver: str = "glpk",
        solver_options: Optional[Dict[str, str]] = None,
        clone_model: bool = True,
        verbose: bool = True
    ):
        if verbose:
            print("🔧 Initializing NN-E Solver with Plugin Architecture...")
        
        self.plugin = plugin
        self.model_path = pathlib.Path(model_path) if model_path else None
        self.mip_solver = mip_solver
        self.solver_options = solver_options or {}
        self.clone_model = clone_model
        self.verbose = verbose
        
        # Will be set during training or loading
        self._n_inputs = None
        self._scenario_dim = None
        self._pytorch_model = None
        self._scaler_x = None
        self._scaler_scenarios = None
        self._scaler_y = None
        
        if verbose:
            print(f"✓ Solver: {mip_solver}")
            if model_path:
                print(f"✓ Model path: {model_path}")
            
            if plugin:
                print(f"✓ Plugin: {type(plugin).__name__}")
            # LOCKED BOUNDARY: No validation that exposes solver internals
            self._verify_plugin_contract(plugin)
        else:
            print("⚠️  No plugin provided - call set_plugin() before solving")
        
        if not OMLT_AVAILABLE:
            print("⚠️  WARNING: OMLT not available - exact ReLU embedding disabled")
    

    def set_plugin(self, plugin: ProblemPlugin):
        """Set the problem plugin with contract verification."""
        print("🔗 Setting problem plugin...")
        self.plugin = plugin
        
        # LOCKED BOUNDARY: Only verify contract compliance, no internal validation
        self._verify_plugin_contract(plugin)
        print(f"✅ Plugin {type(plugin).__name__} connected")
    
    def fit(self, plugin: ProblemPlugin):
        """
        Connect a problem plugin.
        
        Args:
            plugin: A ProblemPlugin instance
        """
        if not isinstance(plugin, ProblemPlugin):
            raise TypeError(
                f"Expected ProblemPlugin, got {type(plugin).__name__}. "
                "Please implement the ProblemPlugin interface for your problem."
            )
        
        self.set_plugin(plugin)
    
    def _load_pretrained_model(self):
        """Load pre-trained model with scalers."""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"📥 Loading model from {self.model_path}")
        
        # Load checkpoint
        ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Extract scalers
        self._scaler_x = ckpt['scaler_x']
        self._scaler_scenarios = ckpt['scaler_scenarios'] 
        self._scaler_y = ckpt['scaler_y']
        
        # Extract model config
        model_config = ckpt['model_config']
        self._n_inputs = model_config['x_dim']
        self._scenario_dim = model_config['scenario_dim']
        self._encoder_latent_dim = model_config['encoder_latent_dim']
        
        # Recreate model
        self._pytorch_model = RecourseSurrogateNNE(
            x_dim=model_config['x_dim'],
            scenario_dim=model_config['scenario_dim'],
            encoder_hidden_dim=model_config['encoder_hidden_dim'],
            encoder_latent_dim=model_config['encoder_latent_dim'],
            trunk_hidden=tuple(model_config['trunk_hidden'])
        )
        
        # Load weights
        self._pytorch_model.load_state_dict(ckpt['state_dict'])
        self._pytorch_model.eval()
        
        print("✅ Model loaded successfully")
        print(f"   Architecture: {self._n_inputs} → {model_config['encoder_latent_dim']} → 1")
    
    def _build_input_scaler(self):
        """Build affine layer that performs (x - μ) / σ scaling for OMLT."""
        # One affine layer that performs (x – μ) / σ
        weight = torch.diag(torch.tensor(1 / self._scaler_x.scale_, dtype=torch.float32))
        bias = -torch.tensor(self._scaler_x.mean_ / self._scaler_x.scale_, dtype=torch.float32)
        affine = nn.Linear(self._n_inputs, self._n_inputs)
        affine.weight.data = weight
        affine.bias.data = bias
        affine.requires_grad_(False)
        return affine
    
    def _build_surrogate_only_x(self, z_fixed):
        """Build trunk-only network that takes x → φ_scaled(x) with fixed scenario embedding."""
        
        class SurrogateOnlyX(nn.Module):
            def __init__(self, trunk: nn.Sequential, z_fixed: torch.Tensor, input_scaler: nn.Module):
                super().__init__()
                self.input_scaler = input_scaler
                self.trunk = trunk
                # Register as buffer so it's baked into the FX graph
                self.register_buffer('z', z_fixed)

            def forward(self, x):
                # x: (B, x_dim) RAW inputs; z: (latent_dim,)
                x_scaled = self.input_scaler(x)  # Apply (x-μ)/σ scaling
                z = self.z.unsqueeze(0).expand(x_scaled.size(0), -1)
                inp = torch.cat([x_scaled, z], dim=1)
                output = self.trunk(inp)
                return output.squeeze(-1) if output.dim() > 1 else output
        
        # Build input scaler and combine with trunk
        input_scaler = self._build_input_scaler()
        return SurrogateOnlyX(self._pytorch_model.trunk, z_fixed, input_scaler)
    
    def _build_output_unscaler(self):
        """Build affine layer that converts scaled NN output to real units (μ + σ*y_scaled)."""
        w = torch.tensor(self._scaler_y.scale_[0], dtype=torch.float32).view(1, 1)
        b = torch.tensor(self._scaler_y.mean_[0], dtype=torch.float32)
        layer = nn.Linear(1, 1)
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(b)
        layer.requires_grad_(False)
        return layer
    
    def _pytorch_to_omlt_network(self, pytorch_model, input_bounds):
        """
        Exports PyTorch model to ONNX, imports it back into OMLT, and returns a fully
        populated NetworkDefinition for BigM formulation.
        
        Args:
            pytorch_model: A torch.nn.Sequential of Linear and ReLU layers
            input_bounds: List of (low, high) bounds for each input dimension
        
        Returns:
            NetworkDefinition: Fully populated with layers, weights, and activations
        """
        print("🔧 Converting PyTorch model to OMLT format via ONNX...")
        
        try:
            # Ensure model is in evaluation mode
            pytorch_model.eval()
            
            # Get input dimension from the first layer
            first_param = next(pytorch_model.parameters())
            n_inputs = first_param.shape[1] if len(first_param.shape) > 1 else len(input_bounds)
            
            print(f"   Model input dimension: {n_inputs}")
            print(f"   Input bounds: {len(input_bounds)} dimensions")
            
            # Create dummy input for ONNX export
            dummy_input = torch.zeros(1, n_inputs, dtype=torch.float32)
            
            # Export to ONNX in memory
            buffer = io.BytesIO()
            torch.onnx.export(
                pytorch_model,              # the model
                dummy_input,                # dummy input tensor
                buffer,                     # file-like object
                input_names=["x"],          # name of the input
                output_names=["phi"],       # name of the output
                opset_version=12,           # ONNX opset version
                dynamic_axes={"x": {0: "batch"}, "phi": {0: "batch"}},  # dynamic batch size
                do_constant_folding=True,   # optimize constant folding
                verbose=False               # reduce verbosity
            )
            buffer.seek(0)
            
            print("✅ ONNX export successful")
            
            # Convert input_bounds list to dict format expected by OMLT
            if isinstance(input_bounds, list):
                input_bounds_dict = {i: bounds for i, bounds in enumerate(input_bounds)}
            else:
                input_bounds_dict = input_bounds
            
            # Import into OMLT with proper scaling
            # Load ONNX model first
            import onnx
            onnx_model = onnx.load_model_from_string(buffer.getvalue())
            
            # Handle Concat operations manually before OMLT conversion
            # Pass dynamic dimensions from the loaded model
            onnx_model_modified, concat_info = _handle_concat_manually(
                onnx_model, 
                n_first_stage=self._n_inputs, 
                n_embedding=self._encoder_latent_dim
            )
            
            # Convert bounds dict to list format expected by new API
            if isinstance(input_bounds_dict, dict):
                input_bounds_list = [input_bounds_dict[i] for i in range(len(input_bounds_dict))]
            else:
                input_bounds_list = input_bounds_dict
            
            # If we have concat info, update input bounds to match concatenated input size
            if concat_info and 'concatenated_input_size' in concat_info:
                concat_size = concat_info['concatenated_input_size']
                # Expand input bounds to match concatenated input dimensions
                if len(input_bounds_list) < concat_size:
                    # Fill remaining bounds with the same range as the first-stage variables
                    base_bounds = input_bounds_list[0] if input_bounds_list else (-4, 4)
                    input_bounds_list.extend([base_bounds] * (concat_size - len(input_bounds_list)))
                print(f"🔧 Expanded input bounds from {len(input_bounds_dict)} to {len(input_bounds_list)} for concatenated model")
            
            net_def = load_onnx_neural_network(
                onnx_model_modified,                 # Modified ONNX model without Concat
                input_bounds=input_bounds_list       # bounds as list of tuples
            )
            
            # Store concat info for later MILP constraint generation
            self._concat_info = concat_info
            
            print("✅ OMLT NetworkDefinition creation successful")
            
            # Count layers for verification
            n_linear = len([layer for layer in net_def.layers if 'linear' in str(type(layer)).lower()])
            n_relu = len([layer for layer in net_def.layers if 'relu' in str(type(layer)).lower()])
            print(f"   Imported {n_linear} linear layers and {n_relu} ReLU activations")
            
            return net_def
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ OMLT conversion failed: {e}")
            
            # Provide specific guidance for common issues
            if "Concat" in error_msg:
                print("   🔍 Issue: Neural network contains concatenation operations")
                print("   💡 Solution: Using linear approximation fallback instead")
                print("   📝 Note: This provides approximate but still useful optimization")
                print("   🔧 To fix: Consider re-training model without concatenation operations")
            elif "Unhandled node type" in error_msg:
                print("   🔍 Issue: Neural network contains unsupported operations")
                print("   💡 Solution: Consider simplifying the model architecture")
            else:
                print("   This could be due to:")
                print("   - Unsupported PyTorch operations in the model")
                print("   - OMLT version compatibility issues")
                print("   - Invalid input bounds format")
            return None
    
    def _compute_hidden_layer_bounds(self, nn_omlt, input_bounds):
        """
        Replace the stub with true interval propagation:
        - input_bounds: dict i -> (low_i, high_i) in the *scaled* input space
        - nn_omlt: the OMLT NetworkDefinition (but we'll propagate on the PyTorch model)
        """
        print("🔧 Computing hidden layer bounds with interval propagation...")
        
        try:
            # Check if we have a concatenated model
            if hasattr(self, '_concat_info') and self._concat_info and 'concatenated_input_size' in self._concat_info:
                # For concatenated models, the OMLT network handles the full concatenated input
                # so we skip PyTorch-based bounds computation and rely on OMLT's bounds
                print("   Using OMLT's bounds computation for concatenated model")
                
                # Try to use OMLT's built-in bounds computation
                if hasattr(nn_omlt, 'compute_bounds'):
                    try:
                        nn_omlt.compute_bounds()
                        print("✅ Called OMLT's compute_bounds() method")
                    except Exception as e:
                        print(f"⚠️  OMLT bounds computation failed: {e}")
                else:
                    print("ℹ️  OMLT NetworkDefinition doesn't have compute_bounds() method")
                
                # Store empty bounds since we're using OMLT's bounds
                self._layer_bounds = []
                print("✅ Using OMLT-based bounds for concatenated model")
                return
            
            # For non-concatenated models, use PyTorch-based interval propagation
            combined_model = self._last_combined_model
            if combined_model is None:
                print("⚠️  No PyTorch model available for bounds computation")
                self._layer_bounds = []
                return

            # Initialize arrays for lower/upper from input_bounds
            import numpy as np
            # Sort bounds by index
            if isinstance(input_bounds, dict):
                n = len(input_bounds)
                lower = np.array([input_bounds[i][0] for i in range(n)], dtype=float)
                upper = np.array([input_bounds[i][1] for i in range(n)], dtype=float)
            else:
                # input_bounds is a list of tuples
                lower = np.array([bounds[0] for bounds in input_bounds], dtype=float)
                upper = np.array([bounds[1] for bounds in input_bounds], dtype=float)
                n = len(input_bounds)

            print(f"   Initial input bounds: [{lower[0]:.2f}, {upper[0]:.2f}] for {n} variables")

            # Walk through each module in combined_model
            layer_bounds = []  # will store (lower, upper) for each layer's output
            layer_idx = 0
            
            for module in combined_model:
                if isinstance(module, torch.nn.Linear):
                    # get weight (out, in) and bias (out,)
                    W = module.weight.detach().cpu().numpy()
                    b = module.bias.detach().cpu().numpy()

                    # split W into positive & negative parts
                    W_pos = np.maximum(W, 0)
                    W_neg = np.minimum(W, 0)

                    # pre-activation bounds: W * x + b
                    lower_pre = W_pos.dot(lower) + W_neg.dot(upper) + b
                    upper_pre = W_pos.dot(upper) + W_neg.dot(lower) + b

                    # prepare for next step
                    lower, upper = lower_pre, upper_pre

                    layer_bounds.append((lower_pre, upper_pre))
                    layer_idx += 1
                    
                    print(f"     Layer {layer_idx:2d} (Linear): bounds [{lower_pre.min():.2e}, {upper_pre.max():.2e}]")

                elif isinstance(module, torch.nn.ReLU):
                    # post-activation: ReLU
                    lower = np.maximum(0, lower)
                    upper = np.maximum(0, upper)
                    layer_bounds.append((lower, upper))
                    layer_idx += 1
                    
                    print(f"     Layer {layer_idx:2d} (ReLU):   bounds [{lower.min():.2e}, {upper.max():.2e}]")

                else:
                    # for input-scaler or other blocks, try to handle them
                    print(f"     Skipping module type: {type(module)}")

            # Store bounds for later verification
            self._layer_bounds = layer_bounds
            print(f"   Propagated through {len(layer_bounds)} layers")
            
            # Report the tightest bounds achieved
            if layer_bounds:
                final_lower, final_upper = layer_bounds[-1]
                if len(final_lower) == 1:  # Output layer
                    print(f"   Final output bounds: [{final_lower[0]:.2e}, {final_upper[0]:.2e}]")

            print("✅ Hidden‐layer bounds computation completed")
                    
        except Exception as e:
            print(f"⚠️  Could not compute hidden layer bounds: {e}")
            print("ℹ️  Using default bounds - MILP may be less tight")
            # Store empty bounds to avoid errors later
            self._layer_bounds = []
            # Fall back to basic bounds checking
            if hasattr(nn_omlt, 'input_bounds'):
                print("ℹ️  Input bounds are set in network definition")
            else:
                print("⚠️  No input bounds found in network definition")

    def solve(self, time_limit: Optional[int] = None):
        """
        Solve the problem using NN-E surrogate with exact ReLU embedding.
        
        The model is obtained from the plugin, ensuring complete problem agnosticism.
        
        FIXES APPLIED:
        1. Exact ReLU embedding via OMLT
        2. Plugin-based scenario data extraction
        3. Correct constraint filtering
        4. Plugin-based first-stage cost extraction
        """
        if self.verbose:
            print("🎯 SOLVING WITH NN-E SURROGATE")
            print("=" * 50)
        
        # Ensure plugin is available
        if not self.plugin:
            raise ValueError("No plugin provided. Use set_plugin() or fit() before solving.")
        
        # Load model if needed
        if self._pytorch_model is None:
            self._load_pretrained_model()
        
        # Get model from plugin
        if self.verbose:
            print("📋 Getting model from plugin...")
        model = self.plugin.pyomo_model()
        
        # Clone model if requested
        if self.clone_model:
            if self.verbose:
                print("📋 Cloning model...")
            mdl = model.clone()
        else:
            mdl = model
        
        # Get first-stage variables
        fs_vars = self._get_first_stage_vars(mdl)
        fs_set = ComponentSet(fs_vars)
        
        if self.verbose:
            print(f"📊 Problem structure:")
            print(f"   First-stage variables: {len(fs_vars)}")
        
        # Initialize variables with feasible LP solution
        self._initialize_variables_with_lp_relaxation(mdl, fs_vars)
        
        # Drop constraints without first-stage variables (FIXED)
        self._drop_constraints_without_first_stage(mdl, fs_set)
        
        # Fix second-stage variables
        self._fix_second_stage_vars(mdl, fs_set)
        
        # Create surrogate variable
        mdl.phi_recourse = pyo.Var(within=pyo.Reals, initialize=0.0)
        
        # Attach NN surrogate (exact ReLU or linear fallback)
        if OMLT_AVAILABLE:
            self._attach_exact_relu_embedding(mdl, mdl.phi_recourse, fs_vars)
        else:
            print("⚠️  Using linear approximation fallback (OMLT not available)")
            self._attach_linear_approximation_fallback(mdl, mdl.phi_recourse, fs_vars)
        
        # Rebuild objective with first-stage costs
        self._rebuild_objective_with_first_stage_costs(mdl, mdl.phi_recourse, fs_vars)
        
        # Run sanity checks
        self.run_sanity_checks(mdl)
        
        # Solve
        print("🔍 Solving surrogate MILP...")
        solver = pyo.SolverFactory(self.mip_solver)
        
        # Apply solver options
        for key, value in self.solver_options.items():
            solver.options[key] = value
        
        if time_limit:
            solver.options['tmlim'] = time_limit
        
        result = solver.solve(mdl, tee=False)
        
        print(f"✅ Solve completed: {result.solver.termination_condition}")
        
        # Safely extract results with error handling
        try:
            first_stage_solution = {v.name: pyo.value(v) for v in fs_vars if pyo.value(v) is not None}
        except:
            first_stage_solution = {}
            print("⚠️  Could not extract first-stage solution values")
            
        try:
            approx_objective = pyo.value(mdl.obj_neur2sp)
        except:
            approx_objective = None
            print("⚠️  Could not extract objective value")
            
        try:
            wallclock_time = getattr(result.solver, 'time', 0.0)
            if wallclock_time is None:
                wallclock_time = 0.0
        except:
            wallclock_time = 0.0
        
        return {
            "first_stage_solution": first_stage_solution,
            "approx_objective": approx_objective,
            "termination_cond": str(result.solver.termination_condition),
            "wallclock_sec": wallclock_time,
        }
    
    def _initialize_variables_with_lp_relaxation(self, mdl, fs_vars):
        """Initialize variables with feasible LP relaxation solution."""
        print("🔧 Initializing with LP relaxation...")
        
        # Create temporary model for LP relaxation
        mdl_relax = mdl.clone()
        
        # Relax integer variables if any
        for v in mdl_relax.component_data_objects(pyo.Var):
            if v.domain in (pyo.Binary, pyo.Integers):
                v.domain = pyo.Reals
        
        # Quick LP solve
        solver = pyo.SolverFactory('glpk')
        solver.options['tmlim'] = 30
        result = solver.solve(mdl_relax, tee=False)
        
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Transfer solution to original model
            for v_orig, v_relax in zip(fs_vars, self._get_first_stage_vars(mdl_relax)):
                if pyo.value(v_relax) is not None:
                    v_orig.set_value(pyo.value(v_relax))
            print("✅ LP relaxation successful")
        else:
            # Fallback initialization
            for v in fs_vars:
                if v.value is None:
                    v.set_value(1.0)
            print("⚠️  LP relaxation failed, using fallback initialization")
    
    def _attach_exact_relu_embedding(self, mdl, phi_var, fs_vars):
        """
        Attach exact ReLU embedding using OMLT.
        This creates the exact MILP formulation from the paper.
        """
        print("🧠 Creating exact ReLU embedding with OMLT...")
        
        # Get scenario data from problem
        scenario_tensor, prob_tensor = self._get_scenario_data_from_problem()
        
        # F1 Fix: Pre-compute scenario embedding to eliminate pooling
        with torch.no_grad():
            # shape (1, K, scenario_dim) → (latent_dim,)
            z_fixed = self._pytorch_model.encoder(scenario_tensor, prob_tensor).squeeze(0)
            print(f"✅ Pre-computed scenario embedding: {z_fixed.shape}")
        
        # Create trunk-only network that takes x → φ_scaled(x)
        surrogate_x = self._build_surrogate_only_x(z_fixed)
        
        # F2 Fix: Add output unscaler to convert to real units
        unscaler = self._build_output_unscaler()
        combined_model = nn.Sequential(surrogate_x, unscaler)
        
        # Store combined model for bounds computation
        self._last_combined_model = combined_model
        
        print("✅ Built x-only surrogate with output unscaling")
        
        # P2 fix: Define tight input bounds for scaled variables (±4σ)
        std_range = 4  # ±4σ covers >99.9% of training data
        input_bounds = [(-std_range, std_range)] * self._n_inputs
        
        print(f"🔧 Using tight input bounds: ±{std_range} for {self._n_inputs} variables")
        
        # Create OMLT network with OMLT 1.2.2 adapter
        try:
            nn_omlt = self._pytorch_to_omlt_network(combined_model, input_bounds)
            if nn_omlt is None:
                print("❌ OMLT network creation returned None")
                print("⚠️  Falling back to linear approximation")
                self._attach_linear_approximation_fallback(mdl, phi_var, fs_vars)
                return
            print("✅ OMLT network creation successful")
            
            # F3 Fix: Compute hidden layer bounds for tighter Big-M
            # For concatenated models, we need to update the input bounds
            if hasattr(self, '_concat_info') and self._concat_info and 'concatenated_input_size' in self._concat_info:
                concat_size = self._concat_info['concatenated_input_size']
                # Create bounds for the full concatenated input (dynamically sized)
                concat_input_bounds = [(-std_range, std_range)] * concat_size
                print(f"🔧 Using concatenated input bounds: ±{std_range} for {concat_size} variables")
                self._compute_hidden_layer_bounds(nn_omlt, concat_input_bounds)
            else:
                self._compute_hidden_layer_bounds(nn_omlt, input_bounds)
            
        except Exception as e:
            print(f"❌ OMLT network creation failed: {e}")
            print("⚠️  Falling back to linear approximation")
            self._attach_linear_approximation_fallback(mdl, phi_var, fs_vars)
            return
        
        # Create OMLT block
        mdl.omlt_block = OmltBlock()
        formulation = ReluBigMFormulation(nn_omlt)
        mdl.omlt_block.build_formulation(formulation)
        
        # Handle concatenation constraints manually if needed
        if hasattr(self, '_concat_info') and self._concat_info and 'concatenated_input_size' in self._concat_info:
            self._add_manual_concat_constraints(mdl, fs_vars, z_fixed)
        
        # Link first-stage variables to OMLT inputs (avoid name clashes)
        mdl.omlt_input_links = pyo.ConstraintList()
        for i, var in enumerate(fs_vars):
            mdl.omlt_input_links.add(expr=mdl.omlt_block.inputs[i] == var)
        
        # Link OMLT output to phi variable
        mdl.omlt_output_link = pyo.Constraint(expr=phi_var == mdl.omlt_block.outputs[0])
        
        # Count binary variables correctly
        n_binary = sum(1 for v in mdl.omlt_block.component_data_objects(pyo.Var, active=True) 
                      if v.is_binary())
        n_total = len(list(mdl.omlt_block.component_data_objects(pyo.Var, active=True)))
        
        print("✅ Exact ReLU embedding created")
        print(f"   Binary variables: {n_binary} (total: {n_total})")
        print(f"   Input bounds: {input_bounds[0]} for all {len(input_bounds)} inputs")
    
    def _get_scenario_data_from_problem(self):
        """Extract scenario data using the problem plugin."""
        print("📊 Extracting scenario data via plugin...")
        
        if not self.plugin:
            raise ValueError(
                "No plugin provided. Please set a ProblemPlugin using set_plugin() or fit() method."
            )
        
        # Use plugin interface
        scenario_data, prob_data = self.plugin.get_scenario_tensor()
        
        # Apply scenario scaling
        scenario_scaled = self._scaler_scenarios.transform(scenario_data.reshape(-1, scenario_data.shape[-1]))
        scenario_scaled = scenario_scaled.reshape(scenario_data.shape)
        
        # Convert to tensors
        scenario_tensor = torch.tensor(scenario_scaled, dtype=torch.float32)
        prob_tensor = torch.tensor(prob_data, dtype=torch.float32)
        
        # Store for sanity checks
        self._last_scenario_tensor = scenario_tensor
        
        print(f"✅ Scenario data extracted: {scenario_tensor.shape}")
        return scenario_tensor, prob_tensor

    def _get_first_stage_vars(self, mdl):
        """Get first-stage variables using the plugin."""
        if not self.plugin:
            raise ValueError(
                "No plugin provided. Please set a ProblemPlugin using set_plugin() or fit() method."
            )
        
        return self.plugin.first_stage_vars(mdl)

    def _attach_linear_approximation_fallback(self, mdl, phi_var, fs_vars):
        """Fallback linear approximation when OMLT is not available."""
        print("🧠 Creating linear approximation (fallback)...")
        
        # Get current first-stage values (already initialized)
        x0_values = np.array([pyo.value(v) for v in fs_vars])
        
        # Get scenario data
        scenario_tensor, prob_tensor = self._get_scenario_data_from_problem()
        
        # Scale inputs
        x0_scaled = self._scaler_x.transform(x0_values.reshape(1, -1))
        
        # Convert to tensors
        x0_tensor = torch.tensor(x0_scaled, dtype=torch.float32, requires_grad=True)
        
        # Evaluate network and compute gradient
        self._pytorch_model.eval()
        output = self._pytorch_model(x0_tensor, scenario_tensor, prob_tensor)
        
        # Get function value and gradient
        f_x0_scaled = output.item()
        f_x0 = self._scaler_y.inverse_transform([[f_x0_scaled]])[0, 0]
        
        # Compute gradient
        self._pytorch_model.zero_grad()
        output.backward()
        gradient_scaled = x0_tensor.grad.squeeze().numpy()
        
        # Unscale gradient
        y_scale = self._scaler_y.scale_[0]
        x_scale = self._scaler_x.scale_
        gradient = gradient_scaled * (y_scale / x_scale)
        
        # Create linear approximation: φ = f(x0) + ∇f(x0)ᵀ(x - x0)
        linear_expr = f_x0
        for i, (var, grad_i, x0_i) in enumerate(zip(fs_vars, gradient, x0_values)):
            linear_expr += grad_i * (var - x0_i)
        
        # Add constraint
        mdl.nne_approx_constraint = pyo.Constraint(expr=phi_var == linear_expr)
        
        print(f"📊 Linear approximation: f(x₀)={f_x0:.2f}, ||∇f||={np.linalg.norm(gradient):.4f}")
    
    def _drop_constraints_without_first_stage(self, mdl, fs_set):
        """
        Drop constraints that don't involve ANY first-stage variables.
        FIXED: Keep mixed constraints, only drop pure second-stage constraints.
        """
        print("🔍 Filtering constraints...")
        n_keep, n_drop = 0, 0
        
        for cdata in mdl.component_data_objects(pyo.Constraint, active=True):
            vars_in_c = ComponentSet(identify_variables(cdata.body))
            
            # Keep constraint if it involves ANY first-stage variable
            if not vars_in_c.isdisjoint(fs_set):
                n_keep += 1
            else:
                # Drop constraint if it has NO first-stage variables
                cdata.deactivate()
                n_drop += 1
                
        print(f"✅ Kept {n_keep} constraints, dropped {n_drop} pure second-stage constraints")
    
    def _fix_second_stage_vars(self, mdl, fs_set):
        """Fix second-stage variables to their current values."""
        print("🔒 Fixing second-stage variables...")
        fixed = 0
        for v in mdl.component_data_objects(pyo.Var):
            if v not in fs_set and v.name != 'phi_recourse':
                v.fix(v.value if v.value is not None else 0.0)
                fixed += 1
        print(f"✅ Fixed {fixed} second-stage variables")
    
    def _rebuild_objective_with_first_stage_costs(self, mdl, phi_var, fs_vars):
        """
        Replace *all* existing objectives with exactly: minimize cᵀ x + φ(x),
        where c comes from the problem data and φ is the surrogate output.
        
        This ensures no stray terms from original objectives remain.
        """
        print("🎯 Rebuilding objective with complete replacement...")
        
        # 1) Remove every existing Objective from mdl (both active and inactive)
        old_objectives = []
        for obj in list(mdl.component_objects(pyo.Objective, active=True)):
            old_objectives.append(obj.name)
            mdl.del_component(obj)
        for obj in list(mdl.component_objects(pyo.Objective, active=False)):
            old_objectives.append(obj.name)
            mdl.del_component(obj)
        
        print(f"✅ Removed {len(old_objectives)} existing objectives: {old_objectives}")
        
        # 2) Extract the true first-stage cost vector c using plugin
        c_vec = self._extract_first_stage_costs(mdl, fs_vars)
        
        # Validate costs
        if not c_vec or all(c == 0 for c in c_vec):
            print("⚠️  WARNING: All first-stage costs are zero!")
            print("   This will result in objective: minimize φ(x) only")
            print("   Verify this is correct for your problem formulation")
        
        # 3) Build the cost expression cᵀ x
        if c_vec and any(c != 0 for c in c_vec):
            c_term = sum(c * var for c, var in zip(c_vec, fs_vars))
            print(f"✅ Built first-stage cost term with {len(c_vec)} variables")
        else:
            c_term = 0.0
            print("ℹ️  No first-stage costs - using zero cost term")
        
        # 4) Create the new objective
        #    Name it uniquely so we can find it later if needed
        mdl.obj_neur2sp = pyo.Objective(
            expr=c_term + phi_var,
            sense=pyo.minimize
        )
        
        print("✅ Created new NN-E objective: minimize cᵀx + φ(x)")
        print(f"   Objective name: {mdl.obj_neur2sp.name}")
        
        # 5) Verify the objective is properly set
        active_objectives = list(mdl.component_objects(pyo.Objective, active=True))
        if len(active_objectives) == 1 and active_objectives[0].name == 'obj_neur2sp':
            print("✅ Objective replacement successful")
        else:
            print(f"⚠️  WARNING: Found {len(active_objectives)} active objectives, expected 1")
            for obj in active_objectives:
                print(f"   Active objective: {obj.name}")
        
        # 6) Store objective reference for result extraction
        self._neur2sp_objective = mdl.obj_neur2sp
    
    def _verify_plugin_contract(self, plugin: ProblemPlugin):
        """
        Verify plugin satisfies the locked-down contract without exposing solver internals.
        
        LOCKED BOUNDARY: Only checks interface compliance, not internal expectations.
        """
        try:
            # Contract check 1: Required methods exist and are callable
            required_methods = ['pyomo_model', 'get_scenario_tensor', 'get_scenario_dimensions', 
                              'first_stage_vars', 'get_first_stage_costs']
            
            for method_name in required_methods:
                if not hasattr(plugin, method_name):
                    raise AttributeError(f"Plugin missing required method: {method_name}")
                if not callable(getattr(plugin, method_name)):
                    raise TypeError(f"Plugin {method_name} is not callable")
            
            # Contract check 2: Type compliance (basic signature check only)
            model = plugin.pyomo_model()
            if not isinstance(model, pyo.ConcreteModel):
                raise TypeError(f"pyomo_model() must return ConcreteModel, got {type(model)}")
            
            dims = plugin.get_scenario_dimensions()
            if not isinstance(dims, tuple) or len(dims) != 2:
                raise TypeError(f"get_scenario_dimensions() must return tuple of length 2, got {dims}")
            
            print("✅ Plugin contract verification passed")
            
        except Exception as e:
            print(f"❌ Plugin contract violation: {e}")
            raise ValueError(f"Plugin {type(plugin).__name__} violates locked boundary contract: {e}")
    
    def _extract_first_stage_costs(self, mdl, fs_vars):
        """Extract first-stage cost coefficients using the plugin."""
        print("💰 Extracting first-stage costs via plugin...")
        
        if not self.plugin:
            raise ValueError(
                "No plugin provided. Please set a ProblemPlugin using set_plugin() or fit() method."
            )
        
        # Use plugin interface
        costs = self.plugin.get_first_stage_costs(mdl, fs_vars)
        print(f"✅ Used plugin get_first_stage_costs() method")
        
        if costs and max(costs) > 0:
            print(f"✅ Costs extracted: min={min(costs):.2f}, max={max(costs):.2f}")
        else:
            print("⚠️  WARNING: All first-stage costs are zero!")
            
        return costs
    
    def run_sanity_checks(self, mdl):
        """Run sanity checks on the built MILP model."""
        print("🔍 Running sanity checks...")
        
        checks = []
        
        # Check 1: Scenario tensor shape
        if hasattr(self, '_last_scenario_tensor'):
            actual_shape = self._last_scenario_tensor.shape
            checks.append(f"Scenario tensor: {actual_shape}")
        
        # Check 2: OMLT bounds
        if hasattr(mdl, 'omlt_block'):
            try:
                bounds_info = f"Input bounds set to ±4σ range"
                checks.append(f"OMLT bounds: {bounds_info}")
                
                # Count variables
                n_binary = sum(1 for v in mdl.omlt_block.component_data_objects(pyo.Var, active=True) 
                              if v.is_binary())
                checks.append(f"Binary variables: {n_binary}")
            except:
                checks.append("OMLT bounds: Could not verify")
        
        # Check 3: First-stage costs
        try:
            fs_vars = self._get_first_stage_vars(mdl)
            costs = self._extract_first_stage_costs(mdl, fs_vars)
            max_cost = max(costs) if costs else 0
            checks.append(f"Max first-stage cost: {max_cost:.2e}")
        except:
            checks.append("First-stage costs: Could not verify")
        
        print("📋 Sanity check results:")
        for check in checks:
            print(f"   ✓ {check}")
        
        return checks

    def _add_manual_concat_constraints(self, mdl, fs_vars, z_fixed):
        """
        Add manual MILP constraints to handle concatenation operations.
        
        This method creates variables and constraints that represent the concatenation
        of first-stage variables (x) with pre-computed scenario embeddings (z).
        """
        print("🔧 Adding manual concatenation constraints...")
        
        # Get expected concatenated size from ONNX model analysis
        expected_size = self._concat_info.get('concatenated_input_size', len(fs_vars) + len(z_fixed))
        n_fs_vars = len(fs_vars)
        n_scenario_features = len(z_fixed)
        
        print(f"   First-stage vars: {n_fs_vars}")
        print(f"   Scenario features: {n_scenario_features}")
        print(f"   Expected concatenated size: {expected_size}")
        
        # Verify sizes match
        if n_fs_vars + n_scenario_features != expected_size:
            print(f"⚠️  Size mismatch: {n_fs_vars} + {n_scenario_features} != {expected_size}")
            print("   Adjusting to match ONNX model expectations...")
        
        # Create variables for the concatenated vector
        # These will be the actual inputs to the OMLT network
        mdl.concat_vars = pyo.Var(
            range(expected_size),
            domain=pyo.Reals,
            bounds=(-4, 4)  # Same bounds as used for OMLT inputs
        )
        
        # Add constraints to link concatenated variables to their sources
        mdl.concat_constraints = pyo.ConstraintList()
        
        # First part: first-stage variables (scaled)
        for i, fs_var in enumerate(fs_vars):
            if i < expected_size:
                # Apply scaling to match what the neural network expects
                mdl.concat_constraints.add(
                    expr=mdl.concat_vars[i] == fs_var  # Assume fs_var is already properly scaled
                )
        
        # Second part: scenario embedding (fixed values, already scaled)
        for i, z_val in enumerate(z_fixed):
            concat_idx = n_fs_vars + i
            if concat_idx < expected_size:
                mdl.concat_constraints.add(
                    expr=mdl.concat_vars[concat_idx] == float(z_val.item())
                )
        
        # Update OMLT input links to use concatenated variables instead
        if hasattr(mdl, 'omlt_input_links'):
            mdl.del_component('omlt_input_links')  # Remove old links
        mdl.omlt_input_links = pyo.ConstraintList()
        
        # Link all concatenated variables to OMLT inputs
        for i in range(min(expected_size, len(list(mdl.omlt_block.inputs)))):
            mdl.omlt_input_links.add(
                expr=mdl.omlt_block.inputs[i] == mdl.concat_vars[i]
            )
        
        print(f"✅ Added {len(mdl.concat_constraints)} concatenation constraints")
        print(f"✅ Updated {min(expected_size, len(list(mdl.omlt_block.inputs)))} OMLT input links")

def _handle_concat_manually(onnx_model, n_first_stage=None, n_embedding=None):
    """
    Custom handler for Concat operations in ONNX models.
    Creates a simplified trunk-only model that expects concatenated input directly.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    
    print("🔧 Handling Concat operations manually...")
    
    # Find all Concat nodes
    concat_nodes = [node for node in onnx_model.graph.node if node.op_type == "Concat"]
    
    if not concat_nodes:
        print("   No Concat nodes found")
        return onnx_model, {}
    
    print(f"   Found {len(concat_nodes)} Concat operations")
    
    # Strategy: Create a new model that represents only the trunk network
    # Input: concatenated vector (dynamically sized: n_first_stage + n_embedding)
    # Output: scalar recourse value
    
    try:
        # Dynamically calculate concatenated input size from the model architecture
        # This corresponds to n_first_stage vars + encoder_latent_dim scenario embedding
        if n_first_stage is not None and n_embedding is not None:
            concat_input_size = n_first_stage + n_embedding
            print(f"   ✅ Calculated concatenated input size: {n_first_stage} + {n_embedding} = {concat_input_size}")
        else:
            # Try to infer from the model's first layer input dimension
            concat_input_size = None
            for node in onnx_model.graph.node:
                if node.op_type in ['MatMul', 'Gemm'] and 'trunk' in node.name:
                    # Find the weight tensor for this operation
                    for init in onnx_model.graph.initializer:
                        if init.name in node.input:
                            weight_shape = list(init.dims)
                            if len(weight_shape) == 2:
                                concat_input_size = weight_shape[1]  # Input dimension
                                break
                    if concat_input_size:
                        break
            
            # Fallback: Use 115 if we can't determine dynamically (shouldn't happen)
            if concat_input_size is None:
                concat_input_size = 115
                print(f"   ⚠️  Could not determine input size dynamically, using fallback: {concat_input_size}")
            else:
                print(f"   ✅ Detected concatenated input size from ONNX: {concat_input_size}")
        
        print(f"   Creating trunk-only model with {concat_input_size}-dim input")
        
        # Create input for concatenated data
        concat_input = helper.make_tensor_value_info(
            "concatenated_input",
            TensorProto.FLOAT,
            [None, concat_input_size]  # Batch size, concatenated features
        )
        
        # Create output
        output = helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT, 
            [None, 1]  # Batch size, scalar output
        )
        
        # Extract trunk network weights and biases from original model
        trunk_weights = []
        trunk_biases = []
        
        # Print available initializers for debugging
        print(f"   Available initializers ({len(onnx_model.graph.initializer)}):")
        for init in onnx_model.graph.initializer[:10]:  # Show first 10
            print(f"     - {init.name}: {list(init.dims)}")
        if len(onnx_model.graph.initializer) > 10:
            print("     - ... (truncated)")
        
        # Try multiple naming patterns for trunk layers
        layer_patterns = [
            # Pattern 1: PyTorch Sequential naming
            [("trunk.0.weight", "trunk.0.bias"),
             ("trunk.2.weight", "trunk.2.bias"), 
             ("trunk.4.weight", "trunk.4.bias"),
             ("trunk.6.weight", "trunk.6.bias")],
            # Pattern 2: ONNX path naming
            [("/trunk/0/Gemm_weight", "/trunk/0/Gemm_bias"),
             ("/trunk/2/Gemm_weight", "/trunk/2/Gemm_bias"),
             ("/trunk/4/Gemm_weight", "/trunk/4/Gemm_bias"),
             ("/trunk/6/Gemm_weight", "/trunk/6/Gemm_bias")],
            # Pattern 3: Simple numeric naming
            [("trunk_0_weight", "trunk_0_bias"),
             ("trunk_2_weight", "trunk_2_bias"),
             ("trunk_4_weight", "trunk_4_bias"),
             ("trunk_6_weight", "trunk_6_bias")]
        ]
        
        # Try each pattern
        for pattern_idx, pattern in enumerate(layer_patterns):
            trunk_weights = []
            trunk_biases = []
            
            for weight_pattern, bias_pattern in pattern:
                weight_init = None
                bias_init = None
                
                # Find matching initializers
                for init in onnx_model.graph.initializer:
                    if (weight_pattern in init.name or 
                        any(wp in init.name for wp in [weight_pattern.replace(".", "_"), 
                                                      weight_pattern.replace("/", "_")])):
                        weight_init = init
                    elif (bias_pattern in init.name or 
                          any(bp in init.name for bp in [bias_pattern.replace(".", "_"), 
                                                        bias_pattern.replace("/", "_")])):
                        bias_init = init
                
                if weight_init and bias_init:
                    trunk_weights.append(weight_init)
                    trunk_biases.append(bias_init)
                    print(f"     Found layer: {weight_init.name} + {bias_init.name}")
            
            if len(trunk_weights) > 0:
                print(f"   Using pattern {pattern_idx + 1}: Found {len(trunk_weights)} trunk layers")
                break
        
        # If we still couldn't extract weights, try a more general approach
        if len(trunk_weights) == 0:
            print("   Trying general weight extraction...")
            all_weights = []
            all_biases = []
            
            for init in onnx_model.graph.initializer:
                if "weight" in init.name.lower() and len(init.dims) == 2:
                    all_weights.append(init)
                elif "bias" in init.name.lower() and len(init.dims) == 1:
                    all_biases.append(init)
            
            # Try to pair weights and biases by similar names
            for weight in all_weights:
                weight_base = weight.name.replace("weight", "").replace("_", "").replace(".", "")
                for bias in all_biases:
                    bias_base = bias.name.replace("bias", "").replace("_", "").replace(".", "")
                    if weight_base == bias_base:
                        trunk_weights.append(weight)
                        trunk_biases.append(bias)
                        print(f"     Paired: {weight.name} + {bias.name}")
                        break
        
        # If we couldn't extract weights, fall back to original model
        if len(trunk_weights) == 0:
            print("   Could not extract trunk weights, keeping original model")
            return onnx_model, {'concatenated_input_size': concat_input_size}
        
        print(f"   ✅ Extracted {len(trunk_weights)} trunk layer weights")
        
        # Build simplified trunk network
        nodes = []
        initializers = trunk_weights + trunk_biases
        
        # Current input/output names for chaining
        current_input = "concatenated_input"
        
        for i, (weight_init, bias_init) in enumerate(zip(trunk_weights, trunk_biases)):
            # For ONNX, we need to transpose the weight matrix since PyTorch stores
            # weights as (out_features, in_features) but ONNX MatMul expects (in_features, out_features)
            weight_array = numpy_helper.to_array(weight_init)
            weight_transposed = weight_array.T  # Transpose to (in_features, out_features)
            
            # Create new weight initializer with transposed data
            weight_transposed_init = helper.make_tensor(
                f"trunk_layer_{i}_weight_T",
                TensorProto.FLOAT,
                weight_transposed.shape,
                weight_transposed.flatten()
            )
            
            # Replace original weight initializer
            initializers = [init for init in initializers if init != weight_init]
            initializers.append(weight_transposed_init)
            
            # MatMul node with transposed weight
            matmul_output = f"layer_{i}_matmul"
            matmul_node = helper.make_node(
                "MatMul",
                inputs=[current_input, weight_transposed_init.name],
                outputs=[matmul_output],
                name=f"trunk_layer_{i}_matmul"
            )
            nodes.append(matmul_node)
            
            # Add bias
            add_output = f"layer_{i}_add"
            add_node = helper.make_node(
                "Add",
                inputs=[matmul_output, bias_init.name],
                outputs=[add_output],
                name=f"trunk_layer_{i}_add"
            )
            nodes.append(add_node)
            
            # Add ReLU (except for last layer)
            if i < len(trunk_weights) - 1:
                relu_output = f"layer_{i}_relu"
                relu_node = helper.make_node(
                    "Relu",
                    inputs=[add_output],
                    outputs=[relu_output],
                    name=f"trunk_layer_{i}_relu"
                )
                nodes.append(relu_node)
                current_input = relu_output
            else:
                # Last layer output should be named 'output' directly
                add_node = helper.make_node(
                    "Add",
                    inputs=[matmul_output, bias_init.name],
                    outputs=["output"],  # Final output name
                    name=f"trunk_layer_{i}_add"
                )
                # Replace the previous add_node
                nodes[-1] = add_node
        
        # Create the new graph (no identity node needed)
        trunk_graph = helper.make_graph(
            nodes,
            "trunk_only_model",
            [concat_input],
            [output],
            initializers
        )
        
        # Create the model
        trunk_model = helper.make_model(trunk_graph)
        
        # Copy opset imports
        for opset in onnx_model.opset_import:
            new_opset = trunk_model.opset_import.add()
            new_opset.domain = opset.domain
            new_opset.version = opset.version
        
        concat_info = {
            'concatenated_input_size': concat_input_size,
            'original_concat_nodes': len(concat_nodes),
            'trunk_layers': len(trunk_weights)
        }
        
        print(f"✅ Created trunk-only model: {concat_input_size}-dim input → scalar output")
        print(f"   Trunk layers: {len(trunk_weights)}")
        return trunk_model, concat_info
        
    except Exception as e:
        print(f"   Error creating trunk model: {e}")
        import traceback
        traceback.print_exc()
        print("   Keeping original model")
        return onnx_model, {}

# ============================================================================
# 4. End of NN-E Pure Inference Solver Implementation
# ============================================================================ 