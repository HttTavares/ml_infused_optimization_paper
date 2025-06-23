#!/usr/bin/env python3
"""
External NN-E Training Module - Completely Decoupled from Solver

This module contains all training-related functionality for NN-E surrogates.
It is completely independent of the solver and can be used standalone for
training neural surrogates on any two-stage stochastic programming problem.

Key principles:
- Complete separation from solver implementation
- Standalone training capabilities
- Generic data format handling
- Reusable across different problems
"""

import json
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. Neural Network Architecture (Extracted from Solver)
# ============================================================================

class DeepSetEncoder(nn.Module):
    """
    Encode an unordered set of scenario feature vectors into a fixed-length
    latent representation using DeepSets architecture.
    
    Each scenario contains:
    - Feature vector (problem-specific)
    - Associated probabilities
    
    FIXED: Probabilities are now injected into φ as input (ξ_k, p_k)
    rather than only used for post-aggregation weighting.
    """
    
    def __init__(
        self,
        scenario_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        agg: str = "sum"  # Switch to sum so outputs represent E[Q]
    ):
        super().__init__()
        # φ now takes (features + probability) as input
        self.phi = nn.Sequential(
            nn.Linear(scenario_dim + 1, hidden_dim),  # +1 for probability
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.agg = agg
    
    def forward(
        self,
        scenarios: torch.Tensor,  # (B, K, scenario_dim)
        probs: Optional[torch.Tensor] = None  # (B, K)
    ) -> torch.Tensor:  # (B, latent_dim)
        assert probs is not None, "DeepSetEncoder now requires probs for NN-E fidelity"
        
        # Build φ inputs: concat feature vector and its probability
        # scenarios: (B, K, D), probs.unsqueeze(-1): (B, K, 1)
        phi_inputs = torch.cat([scenarios, probs.unsqueeze(-1)], dim=-1)  # (B, K, D+1)
        
        # Apply φ to each (ξ_k, p_k) pair
        phi_out = self.phi(phi_inputs)  # (B, K, latent_dim)
        
        # Aggregate across scenarios via simple sum → this now implements the expectation
        if self.agg == "sum":
            return phi_out.sum(dim=1)  # yields E[Ψ1(ξ,p)]
        elif self.agg == "mean":
            return phi_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregation '{self.agg}'")

class RecourseSurrogateNNE(nn.Module):
    """
    NN-E surrogate approximating E[Q(x, Ξ)] for two-stage stochastic programming.
    
    Input:
    - x: first-stage variables 
    - scenarios: encoded scenario features
    - probs: scenario probabilities
    
    Output:
    - predicted expected recourse cost
    """
    
    def __init__(
        self,
        x_dim: int,                          # Number of first-stage variables
        scenario_dim: int,                   # Dimension of each scenario feature vector
        encoder_hidden_dim: int = 128,
        encoder_latent_dim: int = 64,
        trunk_hidden: Tuple[int, ...] = (256, 128, 64)
    ):
        super().__init__()
        
        self.encoder = DeepSetEncoder(
            scenario_dim=scenario_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=encoder_latent_dim
        )
        
        # Trunk network that combines first-stage variables with scenario encoding
        layers = []
        in_dim = x_dim + encoder_latent_dim
        for h in trunk_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.trunk = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,                     # (B, x_dim)
        scenarios: torch.Tensor,             # (B, K, scenario_dim)
        probs: Optional[torch.Tensor] = None # (B, K)
    ) -> torch.Tensor:                       # (B,)
        z = self.encoder(scenarios, probs)   # (B, latent_dim)
        inp = torch.cat([x, z], dim=-1)      # (B, x_dim + latent_dim)
        return self.trunk(inp).squeeze(-1)   # (B,)

# ============================================================================
# 2. Training Dataset (Extracted from Solver)
# ============================================================================

class NNEDataset(Dataset):
    """
    Dataset for training NN-E surrogate on two-stage stochastic programming problems.
    Each sample contains:
    - x: first-stage variables 
    - scenarios: encoded scenario features 
    - probs: scenario probabilities
    - target: expected recourse cost E[Q(x,ξ)]
    
    FIXED: Applies scaling during construction, not during training.
    """
    
    def __init__(self, data_file: str, scaler=None):
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Load raw data
        x_raw = np.array(data['inputs'])
        scenario_raw = np.array(data['scenarios'])  # Shape: (N, K, scenario_dim)
        prob_raw = np.array(data['probabilities'])
        targets_raw = np.array(data['outputs'])
        
        # Setup or use provided scalers
        if scaler is None:
            self.scaler_x = StandardScaler()
            self.scaler_scenarios = StandardScaler()
            self.scaler_y = StandardScaler()
            
            # Fit scalers
            self.scaler_x.fit(x_raw)
            # Flatten scenario data for fitting: (N*K, scenario_dim)
            scenario_flat = scenario_raw.reshape(-1, scenario_raw.shape[-1])
            self.scaler_scenarios.fit(scenario_flat)
            self.scaler_y.fit(targets_raw.reshape(-1, 1))
        else:
            self.scaler_x, self.scaler_scenarios, self.scaler_y = scaler
        
        # Apply scaling and convert to tensors
        x_scaled = self.scaler_x.transform(x_raw)
        scenario_scaled = self.scaler_scenarios.transform(scenario_flat).reshape(scenario_raw.shape)
        targets_scaled = self.scaler_y.transform(targets_raw.reshape(-1, 1)).squeeze()
        
        self.x_data = torch.tensor(x_scaled, dtype=torch.float32)
        self.scenario_data = torch.tensor(scenario_scaled, dtype=torch.float32)
        self.prob_data = torch.tensor(prob_raw, dtype=torch.float32)
        self.targets = torch.tensor(targets_scaled, dtype=torch.float32)
        
        print(f"📊 Dataset loaded with scaling:")
        print(f"   Samples: {len(self.x_data)}")
        print(f"   First-stage dim: {self.x_data.shape[1]}")
        print(f"   Scenario shape: {self.scenario_data.shape[1:]}")
        print(f"   Target range (scaled): [{self.targets.min():.2e}, {self.targets.max():.2e}]")
    
    def get_scalers(self):
        """Return the fitted scalers for later use."""
        return self.scaler_x, self.scaler_scenarios, self.scaler_y
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return (
            self.x_data[idx],      # x
            self.scenario_data[idx], # scenarios
            self.prob_data[idx],   # probs  
            self.targets[idx]      # target
        )

# ============================================================================
# 3. Training Configuration (Extracted from Solver)
# ============================================================================

class TrainConfig:
    """Configuration for NN-E surrogate training."""
    
    def __init__(
        self,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.2,
        early_stopping_patience: int = None,  # Will be set to 0.1 * epochs
        encoder_hidden_dim: int = 128,
        encoder_latent_dim: int = 64,
        trunk_hidden: Tuple[int, ...] = (256, 128, 64)
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.early_stopping_patience = early_stopping_patience or max(1, int(0.1 * epochs))
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_latent_dim = encoder_latent_dim
        self.trunk_hidden = trunk_hidden

# ============================================================================
# 4. Training Function (Extracted from Solver)
# ============================================================================

def train_nne_surrogate(
    model: RecourseSurrogateNNE,
    dataset: NNEDataset,
    config: TrainConfig,
    verbose: bool = True
) -> Tuple[RecourseSurrogateNNE, Dict]:
    """Train the NN-E surrogate with proper validation and early stopping."""
    
    print("🧠 TRAINING NN-E SURROGATE")
    print("=" * 50)
    
    # Device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"🔧 Using device: {device}")
    
    # Split dataset
    n_val = int(len(dataset) * config.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, config.batch_size, shuffle=False)
    
    print(f"📊 Training: {n_train} samples, Validation: {n_val} samples")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    history = {'train': [], 'val': [], 'r2': []}
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for x, scenarios, probs, targets in train_loader:
            # Move to device
            x, scenarios, probs, targets = x.to(device), scenarios.to(device), probs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(x, scenarios, probs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        train_loss /= len(train_ds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true_all, y_pred_all = [], []
        
        if len(val_ds) > 0:  # Only do validation if we have validation data
            with torch.no_grad():
                for x, scenarios, probs, targets in val_loader:
                    x, scenarios, probs, targets = x.to(device), scenarios.to(device), probs.to(device), targets.to(device)
                    predictions = model(x, scenarios, probs)
                    val_loss += criterion(predictions, targets).item() * len(targets)
                    y_true_all.append(targets.cpu())
                    y_pred_all.append(predictions.cpu())
            val_loss /= len(val_ds)
            
            # Calculate R² from validation data
            y_true = torch.cat(y_true_all)
            y_pred = torch.cat(y_pred_all)
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot
        else:
            # No validation data - use training loss and calculate R² on training set
            val_loss = train_loss
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for x, scenarios, probs, targets in train_loader:
                    x, scenarios, probs, targets = x.to(device), scenarios.to(device), probs.to(device), targets.to(device)
                    predictions = model(x, scenarios, probs)
                    y_true_all.append(targets.cpu())
                    y_pred_all.append(predictions.cpu())
            
            y_true = torch.cat(y_true_all)
            y_pred = torch.cat(y_pred_all)
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['r2'].append(r2.item())
        
        # Log every 10 epochs for faster feedback
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | R²: {r2:.3f}")
        
        # Early stopping with deep copy fix
        if val_loss < best_val_loss * 0.999:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.cpu().state_dict())  # FIXED: Deep copy + CPU
            model.to(device)  # Move back to device
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                if verbose:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    if verbose:
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Final R²: {history['r2'][-1]:.3f}")
    
    return model, history

# ============================================================================
# 5. Model Saving/Loading Utilities
# ============================================================================

def save_nne_model(
    model: RecourseSurrogateNNE,
    scalers: Tuple,
    config: TrainConfig,
    save_path: str
):
    """Save complete NN-E model with metadata and scalers."""
    
    scaler_x, scaler_scenarios, scaler_y = scalers
    
    # Extract dimensions from trunk's first layer (more reliable)
    trunk_first_layer = model.trunk[0]  # First layer of trunk
    trunk_input_dim = trunk_first_layer.in_features
    x_dim = trunk_input_dim - config.encoder_latent_dim
    
    save_data = {
        'state_dict': model.cpu().state_dict(),
        'scaler_x': scaler_x,
        'scaler_scenarios': scaler_scenarios,
        'scaler_y': scaler_y,
        'model_config': {
            'x_dim': x_dim,
            'scenario_dim': scaler_scenarios.n_features_in_,  # Get from scaler
            'encoder_hidden_dim': config.encoder_hidden_dim,
            'encoder_latent_dim': config.encoder_latent_dim,
                            'trunk_hidden': config.trunk_hidden
        },
        'solver_metadata': {
            'model_name': 'RecourseSurrogateNNE',
            'problem_type': 'TwoStageStochasticProgram',
            'training_timestamp': datetime.now().isoformat()
        }
    }
    
    torch.save(save_data, save_path)
    print(f"💾 Model and scalers saved to {save_path}")

def create_and_train_nne_model(
    training_data_file: str,
    config_dict: Optional[Dict] = None,
    save_path: str = "trained_nne_surrogate.pt",
    verbose: bool = True
) -> Dict:
    """
    Complete training pipeline for NN-E surrogates.
    
    This is the main external function that replaces the solver's train() method.
    
    Args:
        training_data_file: Path to JSON training data
        config_dict: Training configuration dictionary
        save_path: Path to save trained model
        
    Returns:
        Training results dictionary
    """
    if verbose:
        print("🚀 EXTERNAL NN-E TRAINING PIPELINE")
        print("=" * 50)
    
    # Load dataset
    dataset = NNEDataset(training_data_file)
    
    # Extract dimensions
    sample_x, sample_scenarios, sample_probs, sample_target = dataset[0]
    x_dim = sample_x.shape[0]
    scenario_dim = sample_scenarios.shape[1]  # Features per scenario
    
    if verbose:
        print(f"📊 Model dimensions:")
        print(f"   First-stage variables: {x_dim}")
        print(f"   Scenario dimension: {scenario_dim}")
        print(f"   Max scenarios per sample: {sample_scenarios.shape[0]}")
    
    # Setup training config
    if config_dict is None:
        config_dict = {}
    
    train_config = TrainConfig(
        epochs=config_dict.get('epochs', 200),
        lr=config_dict.get('learning_rate', 1e-3),
        batch_size=config_dict.get('batch_size', 32),
        val_split=config_dict.get('validation_split', 0.2),
        early_stopping_patience=config_dict.get('early_stopping_patience'),
        encoder_hidden_dim=config_dict.get('encoder_hidden_dim', 128),
        encoder_latent_dim=config_dict.get('encoder_latent_dim', 64),
        trunk_hidden=tuple(config_dict.get('trunk_hidden', [256, 128, 64]))
    )
    
    # Create model
    model = RecourseSurrogateNNE(
        x_dim=x_dim,
        scenario_dim=scenario_dim,
        encoder_hidden_dim=train_config.encoder_hidden_dim,
        encoder_latent_dim=train_config.encoder_latent_dim,
        trunk_hidden=train_config.trunk_hidden
    )
    
    if verbose:
        print(f"🏗️  Model architecture:")
        print(f"   Encoder: {scenario_dim} → {train_config.encoder_hidden_dim} → {train_config.encoder_latent_dim}")
        print(f"   Trunk: {x_dim + train_config.encoder_latent_dim} → {train_config.trunk_hidden} → 1")
    
    # Train model - pass verbose flag
    trained_model, history = train_nne_surrogate(model, dataset, train_config, verbose=verbose)
    
    # Save model
    scalers = dataset.get_scalers()
    if verbose:
        save_nne_model(trained_model, scalers, train_config, save_path)
    else:
        # Silent save
        scaler_x, scaler_scenarios, scaler_y = scalers
        trunk_first_layer = model.trunk[0]
        trunk_input_dim = trunk_first_layer.in_features
        x_dim = trunk_input_dim - train_config.encoder_latent_dim
        
        save_data = {
            'state_dict': model.cpu().state_dict(),
            'scaler_x': scaler_x,
            'scaler_scenarios': scaler_scenarios,
            'scaler_y': scaler_y,
            'model_config': {
                'x_dim': x_dim,
                'scenario_dim': scaler_scenarios.n_features_in_,
                'encoder_hidden_dim': train_config.encoder_hidden_dim,
                'encoder_latent_dim': train_config.encoder_latent_dim,
                'trunk_hidden': train_config.trunk_hidden
            },
            'solver_metadata': {
                'model_name': 'RecourseSurrogateNNE',
                'problem_type': 'TwoStageStochasticProgram',
                'training_timestamp': datetime.now().isoformat()
            }
        }
        torch.save(save_data, save_path)
    
    if verbose:
        print(f"💾 Model saved to: {save_path}")
        print(f"✅ External training completed successfully!")
    
    return {
        'history': history,
        'final_r2': history['r2'][-1],
        'best_val_loss': min(history['val']),
        'config': train_config.__dict__,
        'model_path': save_path,
        'scalers': scalers
    } 