import numpy as np
import copy
import json
from typing import Dict, Any, Optional, Tuple

class PerturbFanZhangData:
    # Parameter bounds to ensure realistic values
    PARAM_BOUNDS = {
        'water_coeff': (0.0, 1.0),
        'burning_proportion': (0.0, 1.0),
        'return_to_field_proportion': (0.0, 1.0),
        'probability': (0.01, 1.0),
        'population_x10k': (0.1, None),
        'market_price_CNY_per_kg': (0.01, None),
        'electricity_processing_kWh_per_kg': (0.0, None),
        'straw_to_harvest_ratio': (0.01, None),
        'shoot_to_root_ratio': (0.1, None),
        'root_carbon_content': (0.01, 1.0),
        'emission_coefficient': (0.0, None),
        'cost_CNY_per_kWh': (0.01, None),
        'water_production_function_a': (0.01, None),  # Must be positive
        'water_production_function_b': (None, None),  # Can be negative
        'crop_coefficients': (0.0, 2.0),  # Reasonable range for Kc values
        'water_use_cost': (0.001, None),
        'kwh_per_m3': (0.0, None),
        'unit_cost': (0.01, None),
        'application_rates': (0.1, None),
        'emission_coefficients': (0.0, None),
        'water_availability': (1e6, None),  # Minimum water availability
        'water_delivery_capacity': (1e7, None)  # Minimum WDC
    }
    
    # Perturbation intensity profiles
    PERTURBATION_PROFILES = {
        'mild': {
            'default_R': 0.05,
            'price_R': 0.10,
            'weather_R': 0.15,
            'production_R': 0.05,
            'cost_R': 0.08
        },
        'moderate': {
            'default_R': 0.15,
            'price_R': 0.25,
            'weather_R': 0.30,
            'production_R': 0.15,
            'cost_R': 0.20
        },
        'severe': {
            'default_R': 0.30,
            'price_R': 0.50,
            'weather_R': 0.60,
            'production_R': 0.35,
            'cost_R': 0.40
        }
    }

    def __init__(self, S=3, R=0.05, eps=0.0, kappa=1.0, random_seed=None, 
                 perturbation_profile='mild'):
        self.S = S
        self.R = R
        self.eps = eps
        self.kappa = kappa
        self.perturbation_profile = perturbation_profile
        self.perturbation_metadata = {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get perturbation rates from profile
        if perturbation_profile in self.PERTURBATION_PROFILES:
            self.profile_rates = self.PERTURBATION_PROFILES[perturbation_profile]
        else:
            # Default profile if custom profile name
            self.profile_rates = {
                'default_R': R,
                'price_R': R * 2,
                'weather_R': R * 3,
                'production_R': R,
                'cost_R': R * 1.5
            }

    def perturb(self, base_data):
        """Perturb the data and track metadata"""
        original_data = copy.deepcopy(base_data)
        data = copy.deepcopy(base_data)
        model_data = data['problems']['FanZhangWEFNexusModel']
        
        # Initialize metadata tracking
        self.perturbation_metadata = {
            'profile': self.perturbation_profile,
            'parameters': {
                'S': self.S,
                'R': self.R,
                'eps': self.eps,
                'kappa': self.kappa
            },
            'perturbation_counts': {},
            'original_values': {},
            'perturbed_values': {}
        }

        # 1. Scenario selection/duplication
        hydros = model_data['hydrological_years']
        orig_S = len(hydros)
        if self.S <= orig_S:
            new_hydros = list(np.random.choice(hydros, self.S, replace=False))
        else:
            new_hydros = list(np.random.choice(hydros, self.S, replace=True))
        total_prob = sum(h['probability'] for h in new_hydros)
        for h in new_hydros:
            h['probability'] /= total_prob
        model_data['hydrological_years'] = new_hydros

        # 2. Perturb probabilities with bounds
        original_probs = [h['probability'] for h in model_data['hydrological_years']]
        for h in model_data['hydrological_years']:
            h['probability'] = self._perturb_with_bounds(
                h['probability'], 'probability', 'default_R'
            )
        total_prob = sum(h['probability'] for h in model_data['hydrological_years'])
        for h in model_data['hydrological_years']:
            h['probability'] /= total_prob
        
        self._track_perturbation('scenario_probabilities', original_probs, 
                               [h['probability'] for h in model_data['hydrological_years']])

        # 3. Perturb crop parameters
        for crop in model_data['crops']:
            crop_name = crop['name']
            
            # Market prices (use price_R and kappa)
            orig_price = crop['market_price_CNY_per_kg']
            crop['market_price_CNY_per_kg'] = self._perturb_with_bounds(
                crop['market_price_CNY_per_kg'] * self.kappa, 
                'market_price_CNY_per_kg', 'price_R'
            )
            self._track_perturbation(f'{crop_name}_price', orig_price, 
                                   crop['market_price_CNY_per_kg'])
            
            # Production function parameters
            orig_a = crop['water_production_function']['a']
            orig_b = crop['water_production_function']['b']
            crop['water_production_function']['a'] = self._perturb_with_bounds(
                crop['water_production_function']['a'], 
                'water_production_function_a', 'production_R'
            )
            crop['water_production_function']['b'] = self._perturb_with_bounds(
                crop['water_production_function']['b'], 
                'water_production_function_b', 'production_R'
            )
            self._track_perturbation(f'{crop_name}_production_a', orig_a,
                                   crop['water_production_function']['a'])
            self._track_perturbation(f'{crop_name}_production_b', orig_b,
                                   crop['water_production_function']['b'])
            
            # Other crop parameters
            crop['electricity_processing_kWh_per_kg'] = self._perturb_with_bounds(
                crop['electricity_processing_kWh_per_kg'], 
                'electricity_processing_kWh_per_kg', 'default_R'
            )
            crop['straw_to_harvest_ratio'] = self._perturb_with_bounds(
                crop['straw_to_harvest_ratio'], 'straw_to_harvest_ratio', 'default_R'
            )
            crop['burning_proportion'] = self._perturb_with_bounds(
                crop['burning_proportion'], 'burning_proportion', 'default_R'
            )
            crop['return_to_field_proportion'] = self._perturb_with_bounds(
                crop['return_to_field_proportion'], 'return_to_field_proportion', 'default_R'
            )
            crop['shoot_to_root_ratio'] = self._perturb_with_bounds(
                crop['shoot_to_root_ratio'], 'shoot_to_root_ratio', 'default_R'
            )
            crop['root_carbon_content'] = self._perturb_with_bounds(
                crop['root_carbon_content'], 'root_carbon_content', 'default_R'
            )
            
            # Crop coefficients
            for month, coeff in crop['crop_coefficients'].items():
                crop['crop_coefficients'][month] = self._perturb_with_bounds(
                    coeff, 'crop_coefficients', 'weather_R'
                )
            
            # Emission coefficients
            crop['emission_coefficients']['N2O'] = self._perturb_with_bounds(
                crop['emission_coefficients']['N2O'], 'emission_coefficients', 'default_R'
            )
            crop['straw_burning_emissions_g_per_kg']['CH4'] = self._perturb_with_bounds(
                crop['straw_burning_emissions_g_per_kg']['CH4'], 'emission_coefficients', 'default_R'
            )

        # 4. Perturb irrigation district parameters
        for d in model_data['irrigation_districts']:
            d['population_x10k'] = self._perturb_with_bounds(
                d['population_x10k'], 'population_x10k', 'default_R'
            )
            d['water_coeff']['surface'] = self._perturb_with_bounds(
                d['water_coeff']['surface'], 'water_coeff', 'default_R'
            )
            d['water_coeff']['ground'] = self._perturb_with_bounds(
                d['water_coeff']['ground'], 'water_coeff', 'default_R'
            )

        # 5. Perturb cost parameters
        for k in model_data['water_use_cost_CNY_per_m3']['surface_water']:
            model_data['water_use_cost_CNY_per_m3']['surface_water'][k] = self._perturb_with_bounds(
                model_data['water_use_cost_CNY_per_m3']['surface_water'][k], 'water_use_cost', 'cost_R'
            )
        for k in model_data['water_use_cost_CNY_per_m3']['groundwater']:
            model_data['water_use_cost_CNY_per_m3']['groundwater'][k] = self._perturb_with_bounds(
                model_data['water_use_cost_CNY_per_m3']['groundwater'][k], 'water_use_cost', 'cost_R'
            )

        # 6. Perturb electricity parameters
        irr = model_data['electricity']['irrigation']
        for k in irr['kWh_per_m3_surface']:
            irr['kWh_per_m3_surface'][k] = self._perturb_with_bounds(
                irr['kWh_per_m3_surface'][k], 'kwh_per_m3', 'cost_R'
            )
        for k in irr['kWh_per_m3_ground']:
            irr['kWh_per_m3_ground'][k] = self._perturb_with_bounds(
                irr['kWh_per_m3_ground'][k], 'kwh_per_m3', 'cost_R'
            )
        irr['cost_CNY_per_kWh'] = self._perturb_with_bounds(
            irr['cost_CNY_per_kWh'], 'cost_CNY_per_kWh', 'cost_R'
        )
        irr['emission_coefficient'] = self._perturb_with_bounds(
            irr['emission_coefficient'], 'emission_coefficient', 'default_R'
        )

        # 7. Perturb input application rates
        apps = model_data['input_application_rates']
        for k in apps['unit_cost_CNY']:
            apps['unit_cost_CNY'][k] = self._perturb_with_bounds(
                apps['unit_cost_CNY'][k], 'unit_cost', 'cost_R'
            )
        for k in apps:
            if k.endswith('_kg_per_ha') or k == 'diesel_L_per_ha':
                for kk in apps[k]:
                    apps[k][kk] = self._perturb_with_bounds(
                        apps[k][kk], 'application_rates', 'default_R'
                    )
        for k in apps['emission_coefficients']:
            apps['emission_coefficients'][k] = self._perturb_with_bounds(
                apps['emission_coefficients'][k], 'emission_coefficients', 'default_R'
            )

        # 8. Perturb monthly water availability data (ASWtk and AGWtk) - weather-sensitive
        if 'ASWtk' in model_data:
            for district in model_data['ASWtk']:
                for scenario in model_data['ASWtk'][district]:
                    for month in model_data['ASWtk'][district][scenario]:
                        model_data['ASWtk'][district][scenario][month] = self._perturb_with_bounds(
                            model_data['ASWtk'][district][scenario][month], 
                            'water_availability', 'weather_R'
                        )
        
        if 'AGWtk' in model_data:
            for district in model_data['AGWtk']:
                for scenario in model_data['AGWtk'][district]:
                    for month in model_data['AGWtk'][district][scenario]:
                        model_data['AGWtk'][district][scenario][month] = self._perturb_with_bounds(
                            model_data['AGWtk'][district][scenario][month], 
                            'water_availability', 'weather_R'
                        )

        # 9. Perturb water delivery capacity (WDCi)
        if 'WDCi' in model_data:
            for district in model_data['WDCi']:
                model_data['WDCi'][district] = self._perturb_with_bounds(
                    model_data['WDCi'][district], 'water_delivery_capacity', 'default_R'
                )

        # 10. Apply slack constraints (multiply by (1+eps))
        for d in model_data['irrigation_districts']:
            d['population_x10k'] = d['population_x10k'] * (1 + self.eps)
        
        # Apply slack to monthly water availability
        if 'ASWtk' in model_data:
            for district in model_data['ASWtk']:
                for scenario in model_data['ASWtk'][district]:
                    for month in model_data['ASWtk'][district][scenario]:
                        model_data['ASWtk'][district][scenario][month] *= (1 + self.eps)
        
        if 'AGWtk' in model_data:
            for district in model_data['AGWtk']:
                for scenario in model_data['AGWtk'][district]:
                    for month in model_data['AGWtk'][district][scenario]:
                        model_data['AGWtk'][district][scenario][month] *= (1 + self.eps)

        # Apply slack to water delivery capacity
        if 'WDCi' in model_data:
            for district in model_data['WDCi']:
                model_data['WDCi'][district] *= (1 + self.eps)

        # Calculate perturbation distances
        self.perturbation_metadata['distances'] = self._calculate_perturbation_distances(
            original_data, data
        )

        return data

    def _perturb_with_bounds(self, val: float, param_type: str, rate_type: str) -> float:
        """Perturb a value while respecting parameter bounds"""
        if not isinstance(val, (int, float)):
            return val
        
        # Get perturbation rate from profile
        rate = self.profile_rates.get(rate_type, self.profile_rates['default_R'])
        
        # Apply perturbation
        perturbed_val = float(val) * np.random.uniform(1 - rate, 1 + rate)
        
        # Apply bounds if they exist
        if param_type in self.PARAM_BOUNDS:
            min_bound, max_bound = self.PARAM_BOUNDS[param_type]
            if min_bound is not None:
                perturbed_val = max(perturbed_val, min_bound)
            if max_bound is not None:
                perturbed_val = min(perturbed_val, max_bound)
        
        # Track perturbation count
        if param_type not in self.perturbation_metadata['perturbation_counts']:
            self.perturbation_metadata['perturbation_counts'][param_type] = 0
        self.perturbation_metadata['perturbation_counts'][param_type] += 1
        
        return perturbed_val

    def _track_perturbation(self, param_name: str, original_val: Any, perturbed_val: Any):
        """Track original and perturbed values for key parameters"""
        self.perturbation_metadata['original_values'][param_name] = original_val
        self.perturbation_metadata['perturbed_values'][param_name] = perturbed_val

    def _calculate_perturbation_distances(self, original_data: Dict, perturbed_data: Dict) -> Dict[str, float]:
        """Calculate various distance metrics to quantify perturbation magnitude"""
        distances = {}
        
        orig_model = original_data['problems']['FanZhangWEFNexusModel']
        pert_model = perturbed_data['problems']['FanZhangWEFNexusModel']
        
        # Price distance (economic impact)
        orig_prices = [c['market_price_CNY_per_kg'] for c in orig_model['crops']]
        pert_prices = [c['market_price_CNY_per_kg'] for c in pert_model['crops']]
        distances['price_distance'] = np.mean([abs(p-o)/o for p, o in zip(pert_prices, orig_prices)])
        
        # Water availability distance (resource impact)
        orig_water = []
        pert_water = []
        for district in orig_model['ASWtk']:
            for scenario in orig_model['ASWtk'][district]:
                for month in orig_model['ASWtk'][district][scenario]:
                    orig_water.append(orig_model['ASWtk'][district][scenario][month])
                    pert_water.append(pert_model['ASWtk'][district][scenario][month])
        distances['water_distance'] = np.mean([abs(p-o)/o for p, o in zip(pert_water, orig_water)])
        
        # Production function distance (technological impact)
        orig_prod_a = [c['water_production_function']['a'] for c in orig_model['crops']]
        pert_prod_a = [c['water_production_function']['a'] for c in pert_model['crops']]
        distances['production_distance'] = np.mean([abs(p-o)/o for p, o in zip(pert_prod_a, orig_prod_a)])
        
        # Overall distance (aggregate measure)
        distances['overall_distance'] = np.mean([
            distances['price_distance'],
            distances['water_distance'], 
            distances['production_distance']
        ])
        
        return distances

    def get_perturbation_metadata(self) -> Dict[str, Any]:
        """Return perturbation metadata for experimental analysis"""
        return self.perturbation_metadata

    def save(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def save_with_metadata(self, data, data_filename, metadata_filename):
        """Save both perturbed data and metadata"""
        self.save(data, data_filename)
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(self.perturbation_metadata, f, indent=2)

    def _perturb(self, val):
        """Legacy method for backward compatibility"""
        if isinstance(val, (int, float)):
            return float(val) * np.random.uniform(1 - self.R, 1 + self.R)
        return val 