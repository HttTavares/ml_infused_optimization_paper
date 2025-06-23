import json
from pathlib import Path
from typing import Any, Dict, List, Dict, Any, Optional
from pyomo.environ import *
try:
    from base_problem import BaseProblem
except ImportError:
    from problems.base_problem import BaseProblem

class FanZhangWEFNexusModel(BaseProblem):
    def __init__(self, data_file: str = None, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        self.data: Dict[str, Any] = self.extract_data()

    @property
    def name(self) -> str:
        return "FanZhangWEFNexusModel"

    def build_model(self) -> ConcreteModel:
        m = ConcreteModel()

        # -------------------- basic index sets -------------------- #
        districts = [d["code"] for d in self.data["irrigation_districts"]]
        crops     = [c["name"] for c in self.data["crops"]]
        scen      = [h["name"] for h in self.data["hydrological_years"]]

        # add (optional) monthly index straight from crop-coefficient keys
        months = sorted({mth for c in self.data["crops"] for mth in c["crop_coefficients"]})
        m.I, m.J, m.S, m.T = Set(initialize=districts), Set(initialize=crops), Set(initialize=scen), Set(initialize=months)

        # -------------------- parameters -------------------- #
        admin = {d["code"]: d["admin_district"] for d in self.data["irrigation_districts"]}
        m.prob = Param(m.S, initialize={h["name"]: h["probability"] for h in self.data["hydrological_years"]}, within=NonNegativeReals)

        m.a  = Param(m.J, initialize={c["name"]: c["water_production_function"]["a"]  for c in self.data["crops"]})
        m.b  = Param(m.J, initialize={c["name"]: c["water_production_function"]["b"]  for c in self.data["crops"]})

        # monthly kc  (if a crop lacks a month key we default to 0)
        kc_init = {(c["name"], t): c["crop_coefficients"].get(t, 0)
                   for c in self.data["crops"] for t in months}
        m.kc_t = Param(m.J, m.T, initialize=kc_init, within=NonNegativeReals)
        m.kc = Param(m.J, initialize={j: sum(m.kc_t[j, t] for t in m.T) / len(m.T) for j in m.J})

        m.p          = Param(m.J, initialize={c["name"]: c["market_price_CNY_per_kg"] for c in self.data["crops"]})
        m.proc_kwh   = Param(m.J, initialize={c["name"]: c["electricity_processing_kWh_per_kg"] for c in self.data["crops"]})
        m.straw_ratio= Param(m.J, initialize={c["name"]: c["straw_to_harvest_ratio"] for c in self.data["crops"]})
        m.burn_prop  = Param(m.J, initialize={c["name"]: c["burning_proportion"] for c in self.data["crops"]})
        m.ret_prop   = Param(m.J, initialize={c["name"]: c["return_to_field_proportion"] for c in self.data["crops"]})
        m.emis_CH4_g = Param(m.J, initialize={c["name"]: c["straw_burning_emissions_g_per_kg"]["CH4"] for c in self.data["crops"]})
        m.shoot_root = Param(m.J, initialize={c["name"]: c["shoot_to_root_ratio"] for c in self.data["crops"]})
        m.root_c     = Param(m.J, initialize={c["name"]: c["root_carbon_content"]     for c in self.data["crops"]})
        m.emis_N2O   = Param(m.J, initialize={c["name"]: c["emission_coefficients"]["N2O"] for c in self.data["crops"]})

        conv = self.data["carbon_conversion_coefficients"]
        m.CH4_to_CO2, m.N2O_to_CO2 = Param(initialize=conv["CH4_to_CO2eq"]), Param(initialize=conv["N2O_to_CO2eq"])

        # water-use efficiencies
        eff_surf = {d["code"]: d["water_coeff"]["surface"] for d in self.data["irrigation_districts"]}
        eff_gw   = {d["code"]: d["water_coeff"]["ground"]  for d in self.data["irrigation_districts"]}
        m.eta_surf, m.eta_gw = Param(m.I, initialize=eff_surf), Param(m.I, initialize=eff_gw)

        # land – upper and (new) lower bounds
        pop = {d["code"]: d["population_x10k"] for d in self.data["irrigation_districts"]}
        m.Lmax = Param(m.I, initialize={i: pop[i] * 100 for i in districts})              # original upper bound (ha)
        # ---> 61 m.Lmin = Param(m.I, initialize=lambda m, i: 0.3 * m.Lmax[i])   # 30 % inertia floor
        m.Lmin = Param(m.I, initialize=lambda m, i: 0.3 * m.Lmax[i])   # 30 % inertia floor

        # 1) Monthly climate & supply parameters
        SWmax_init = {}
        GWmax_init = {}
        EP_init    = {}
        for i in districts:
            for s in scen:
                for t in months:
                    SWmax_init[i, s, t] = self.data["ASWtk"][i][s][t]
                    GWmax_init[i, s, t] = self.data["AGWtk"][i][s][t]
                    EP_init[i, t, s] = self.data.get("EPitk", {}).get(i, {}).get(s, {}).get(t, 0.0)

        m.SWmax_t = Param(m.I, m.S, m.T, initialize=SWmax_init, within=NonNegativeReals)
        m.GWmax_t = Param(m.I, m.S, m.T, initialize=GWmax_init, within=NonNegativeReals)
        m.EP      = Param(m.I, m.T, m.S, initialize=EP_init,    within=NonNegativeReals)

        # Legacy seasonal totals (for backward compatibility)
        SW_avail, GW_avail = {}, {}
        for i in districts:
            for s in scen:
                SW_avail[i, s] = sum(SWmax_init[i, s, t] for t in months)
                GW_avail[i, s] = sum(GWmax_init[i, s, t] for t in months)
        m.SWmax = Param(m.I * m.S, initialize=SW_avail)
        m.GWmax = Param(m.I * m.S, initialize=GW_avail)

        # crop net-water requirement coefficient (mm per ha equiv.) + stress bounds
        m.wreq     = Param(m.J, initialize={"wheat": 2, "corn": 3, "vegetables": 5})
        m.theta_min = Param(initialize=0.80)   # 80 % deficit limit
        m.theta_max = Param(initialize=1.20)   # 120 % excess limit

        # economic & energy parameters (unchanged from original)
        surf_price = self.data["water_use_cost_CNY_per_m3"]["surface_water"]
        gw_price   = self.data["water_use_cost_CNY_per_m3"]["groundwater"]
        m.c_surf   = Param(m.I, initialize={i: surf_price[admin[i]] for i in districts})
        m.c_gw     = Param(m.I, initialize={i: gw_price[admin[i]]  for i in districts})

        irr = self.data["electricity"]["irrigation"]
        m.kwh_surf = Param(m.I, initialize={i: irr["kWh_per_m3_surface"][admin[i]] for i in districts})
        m.kwh_gw   = Param(m.I, initialize={i: irr["kWh_per_m3_ground"][admin[i]]  for i in districts})
        m.c_elec   = Param(initialize=irr["cost_CNY_per_kWh"])

        apps   = self.data["input_application_rates"]
        cost_u = apps["unit_cost_CNY"]

        def per_ha(field, key=None):
            k = key or f"{field}_kg_per_ha"
            k = k if field != "diesel" else "diesel_L_per_ha"
            return {i: apps[k][admin[i]] for i in districts}

        m.appl_fert, m.appl_pest = Param(m.I, initialize=per_ha("fertilizer")), Param(m.I, initialize=per_ha("pesticide"))
        m.appl_film, m.appl_dies = Param(m.I, initialize=per_ha("film")),       Param(m.I, initialize=per_ha("diesel"))
        m.c_fert, m.c_pest, m.c_film, m.c_dies = Param(initialize=cost_u["fertilizer_per_kg"]), Param(initialize=cost_u["pesticide_per_kg"]), Param(initialize=cost_u["film_per_kg"]), Param(initialize=cost_u["diesel_per_L"])

        emis = apps["emission_coefficients"]
        m.emis_fert = Param(initialize=emis["fertilizer_kgCO2eq_per_kg"])
        m.emis_pest = Param(initialize=emis["pesticide_kgCO2eq_per_kg"])
        m.emis_film = Param(initialize=emis["film_kgCO2eq_per_kg"])
        m.emis_dies = Param(initialize=emis["diesel_kgCO2eq_per_L"])
        m.elec_emis = Param(initialize=irr["emission_coefficient"])
        m.straw_c   = Param(initialize=0.45)

        # food-security baseline demand (t) – mutable so the user can set a realistic value later
        m.FoodDemand = Param(initialize=0, mutable=True)

        # -------------------- decision variables -------------------- #
        m.Area = Var(m.I, m.J, domain=NonNegativeReals)
        # 2) Replace aggregate irrigation vars with monthly ones
        m.SW_t = Var(m.I, m.J, m.S, m.T, domain=NonNegativeReals)
        m.GW_t = Var(m.I, m.J, m.S, m.T, domain=NonNegativeReals)
        m.Yield= Var(m.I, m.J, m.S, domain=NonNegativeReals)

        # Note: First-stage variables (Area[i,j]) are identified dynamically by first_stage_vars() method

        # Season‐total expressions (so downstream code is unchanged)
        m.SW = Expression(m.I, m.J, m.S,
                          rule=lambda m, i, j, s: sum(m.SW_t[i, j, s, t] for t in m.T))
        m.GW = Expression(m.I, m.J, m.S,
                          rule=lambda m, i, j, s: sum(m.GW_t[i, j, s, t] for t in m.T))

        # -------------------- constraints -------------------- #
        m.land_upper = Constraint(m.I, rule=lambda m, i:
                                  sum(m.Area[i, j] for j in m.J) <= m.Lmax[i])
        m.land_lower = Constraint(m.I, rule=lambda m, i:
                                  sum(m.Area[i, j] for j in m.J) >= m.Lmin[i])

        # 3) Monthly supply‐cap constraints
        m.sw_cap = Constraint(m.I, m.S, m.T,
                              rule=lambda m, i, s, t:
                                sum(m.SW_t[i, j, s, t] for j in m.J) <= m.SWmax_t[i, s, t])
        m.gw_cap = Constraint(m.I, m.S, m.T,
                              rule=lambda m, i, s, t:
                                sum(m.GW_t[i, j, s, t] for j in m.J) <= m.GWmax_t[i, s, t])

        # Legacy seasonal constraints (for backward compatibility with WDC calculation)
        m.sw_con = Constraint(m.I, m.S, rule=lambda m, i, s:
                              sum(m.SW[i, j, s] for j in m.J) <= m.SWmax[i, s])
        m.gw_con = Constraint(m.I, m.S, rule=lambda m, i, s:
                              sum(m.GW[i, j, s] for j in m.J) <= m.GWmax[i, s])

        # 4) Irrigation‐adequacy bounds per month (+EP)
        def w_low_rule(m, i, j, s, t):
            return ( m.eta_surf[i]*m.SW_t[i, j, s, t]
                   + m.eta_gw[i]  *m.GW_t[i, j, s, t]
                   + m.EP[i, t, s]
                   ) >= m.theta_min * m.wreq[j] * m.kc_t[j, t] * m.Area[i, j]
        m.wreq_low = Constraint(m.I, m.J, m.S, m.T, rule=w_low_rule)

        def w_up_rule(m, i, j, s, t):
            return ( m.eta_surf[i]*m.SW_t[i, j, s, t]
                   + m.eta_gw[i]  *m.GW_t[i, j, s, t]
                   + m.EP[i, t, s]
                   ) <= m.theta_max * m.wreq[j] * m.kc_t[j, t] * m.Area[i, j]
        m.wreq_up = Constraint(m.I, m.J, m.S, m.T, rule=w_up_rule)

        # 5) Monthly‐based yield response
        m.yield_con = Constraint(m.I, m.J, m.S,
            rule=lambda m, i, j, s:
                m.Yield[i, j, s]
              == m.a[j] * sum(
                    m.eta_surf[i]*m.SW_t[i, j, s, t]
                  + m.eta_gw[i]  *m.GW_t[i, j, s, t]
                    for t in m.T
                ) * 1000
              + m.b[j] * m.Area[i, j]
        )

        # water-delivery capacity (from JSON data)
        m.WDC = Param(m.I, initialize={i: self.data["WDCi"][i] for i in districts}, within=NonNegativeReals)
        m.water_deliv = Constraint(m.I, m.S, rule=lambda m, i, s:
                                   sum(m.SW[i, j, s] + m.GW[i, j, s] for j in m.J) <= m.WDC[i])

        # food-security (basic cereal demand)
        cereals = ["wheat", "corn"]
        def food_sec_rule(m):
            return sum(m.prob[s] * m.Yield[i, j, s]
                       for i in m.I for j in cereals for s in m.S) >= m.FoodDemand
        m.food_sec = Constraint(rule=food_sec_rule)

        # -------------------- objective pieces -------------------- #
        m.Profit = Expression(rule=lambda m:
            sum(m.prob[s]*m.p[j]*m.Yield[i, j, s]
                for i in m.I for j in m.J for s in m.S)
            - (
                sum(m.prob[s]*(m.c_surf[i]*m.SW[i, j, s] + m.c_gw[i]*m.GW[i, j, s])
                    for i in m.I for j in m.J for s in m.S)
              + sum(m.prob[s]*(m.kwh_surf[i]*m.SW[i, j, s] + m.kwh_gw[i]*m.GW[i, j, s])*m.c_elec
                    for i in m.I for j in m.J for s in m.S)
              + sum(m.prob[s]*m.proc_kwh[j]*m.Yield[i, j, s]*m.c_elec
                    for i in m.I for j in m.J for s in m.S)
              + sum(m.appl_fert[i]*m.Area[i, j]*m.c_fert +
                    m.appl_pest[i]*m.Area[i, j]*m.c_pest +
                    m.appl_film[i]*m.Area[i, j]*m.c_film +
                    m.appl_dies[i]*m.Area[i, j]*m.c_dies
                    for i in m.I for j in m.J)
            ))

        # carbon-footprint sub-expressions (unchanged)
        m.CF_inputs = Expression(rule=lambda m:
            sum(m.appl_fert[i]*m.Area[i, j]*m.emis_fert +
                m.appl_pest[i]*m.Area[i, j]*m.emis_pest +
                m.appl_film[i]*m.Area[i, j]*m.emis_film +
                m.appl_dies[i]*m.Area[i, j]*m.emis_dies
                for i in m.I for j in m.J))
        m.CF_irrig = Expression(rule=lambda m:
            sum(m.prob[s]*(m.kwh_surf[i]*m.SW[i, j, s] + m.kwh_gw[i]*m.GW[i, j, s])*m.elec_emis
                for i in m.I for j in m.J for s in m.S))
        m.CF_proc = Expression(rule=lambda m:
            sum(m.prob[s]*m.proc_kwh[j]*m.Yield[i, j, s]*m.elec_emis
                for i in m.I for j in m.J for s in m.S))
        m.CF_straw = Expression(rule=lambda m:
            sum(m.prob[s]*m.Yield[i, j, s]*m.straw_ratio[j]*m.burn_prop[j]*
                (m.emis_CH4_g[j]/1000)*m.CH4_to_CO2
                for i in m.I for j in m.J for s in m.S))
        m.CF_root = Expression(rule=lambda m:
            -sum(m.prob[s]*(m.Yield[i, j, s]/m.shoot_root[j])*m.root_c[j]*(44/12)
                 for i in m.I for j in m.J for s in m.S))
        m.CF_straw_ret = Expression(rule=lambda m:
            -sum(m.prob[s]*m.Yield[i, j, s]*m.straw_ratio[j]*m.ret_prop[j]*
                 m.straw_c*(44/12)
                 for i in m.I for j in m.J for s in m.S))
        m.CF_N2O = Expression(rule=lambda m:
            sum(m.appl_fert[i]*m.Area[i, j]*m.emis_N2O[j]*m.N2O_to_CO2
                for i in m.I for j in m.J))
        m.CF_total = Expression(rule=lambda m:
            m.CF_inputs + m.CF_irrig + m.CF_proc + m.CF_straw +
            m.CF_root + m.CF_straw_ret + m.CF_N2O)

        # extra outputs for post-solution analysis
        m.TY = Expression(rule=lambda m:
            sum(m.prob[s]*m.Yield[i, j, s] for i in m.I for j in m.J for s in m.S))
        m.WU = Expression(rule=lambda m:
            sum(m.prob[s]*(m.SW[i, j, s] + m.GW[i, j, s])
                for i in m.I for j in m.J for s in m.S))
        # (IWP = TY / WU can be calculated after solving to stay linear)

        # ------------------------------------------------------------------ #
        # 4.  Extra objective pieces                                         #
        # ------------------------------------------------------------------ #
        # Total net-benefit already captured in m.Profit
        # Total yield (t) and total water use (m3) were defined earlier
        #   m.TY  – expected total yield (t)
        #   m.WU  – expected total water use (m³)

        # --- NEW: ecosystem-service value ---------------------------------- #
        m.scc = Param(initialize=0.0, mutable=True,      # ← social cost of carbon
                      doc="CNY per kg CO₂-eq avoided")
        m.ESV = Expression(rule=lambda m: -m.scc * m.CF_total)

        # Proxy for irrigation-water productivity (linear)
        m.beta = Param(initialize=1e-4, within=NonNegativeReals, mutable=True)
        m.IWP_lin = Expression(rule=lambda m: m.TY - m.beta*m.WU)

        # ------------------------------------------------------------------ #
        # 5.  Multi-objective handling                                       #
        # ------------------------------------------------------------------ #

        # ---------------- weighted-sum scalarisation (default) ------------- #
        m.w_profit = Param(initialize=1.0,   mutable=True)
        m.w_cf     = Param(initialize=1e-4,  mutable=True)
        m.w_esv    = Param(initialize=1e-4,  mutable=True)          # ← NEW
        m.w_yield  = Param(initialize=1e-2,  mutable=True)
        m.w_iwp    = Param(initialize=1e-2,  mutable=True)

        m.obj_ws = Objective(
            rule=lambda m: ( m.w_profit*m.Profit          # maximise
                           - m.w_cf   *m.CF_total         # minimise
                           + m.w_esv  *m.ESV              # maximise
                           + m.w_yield*m.TY               # maximise
                           + m.w_iwp  *m.IWP_lin ),       # maximise
            sense=maximize)

        # ---------------- ε-constraint variant (optional) ------------------ #
        m.use_epsilon = Param(initialize=0, within=Binary, mutable=True)

        m.eps_cf    = Param(initialize=1e6,  mutable=True)   # upper bound (kg)
        m.eps_esv   = Param(initialize=-1e6, mutable=True)   # lower bound (CNY)  ← NEW
        m.eps_yield = Param(initialize=0.0,  mutable=True)
        m.eps_iwp   = Param(initialize=-1e6, mutable=True)

        def obj_eps_rule(m):
            return m.Profit + m.ESV                      # primary objective
        m.obj_eps = Objective(rule=obj_eps_rule, sense=maximize)
        m.obj_eps.deactivate()  # Start deactivated

        # Create ε-constraint bounds but keep them deactivated initially
        m.cf_cap  = Constraint(expr=m.CF_total  <= m.eps_cf)
        m.esv_req = Constraint(expr=m.ESV       >= m.eps_esv)
        m.yld_req = Constraint(expr=m.TY        >= m.eps_yield)
        m.iwp_req = Constraint(expr=m.IWP_lin   >= m.eps_iwp)
        m.cf_cap.deactivate()
        m.esv_req.deactivate()
        m.yld_req.deactivate()
        m.iwp_req.deactivate()

        # activate / deactivate - ensure only one objective is active
        if value(m.use_epsilon):
            print('Using epsilon constraint method')
            m.obj_ws.deactivate()
            m.obj_eps.activate()
            m.cf_cap.activate()
            m.esv_req.activate()
            m.yld_req.activate()
            m.iwp_req.activate()
        else:
            # print('Using weighted sum method')
            m.obj_eps.deactivate()
            m.obj_ws.activate()
            m.cf_cap.deactivate()
            m.esv_req.deactivate()
            m.yld_req.deactivate()
            m.iwp_req.deactivate()

        return m

    def initialize_variables_for_neural_network(self, model, verbose=True):
        """
        Initialize first-stage variables with realistic values based on data.
        
        This method uses the JSON data to set meaningful starting values for variables,
        making it particularly useful for neural network-based solving where good
        initialization points improve approximation quality.
        
        NOTE: This is separate from normal Pyomo solving which works fine without
        initialization. Only call this for neural network approaches.
        
        Args:
            model: The Pyomo model to initialize
            verbose: If True, print detailed initialization progress (default: True)
        """
        if verbose:
            print("🌱 Initializing variables with data-driven values for neural network...")
        
        # Extract data structures
        districts_data = {d["code"]: d for d in self.data["irrigation_districts"]}
        crops_data = {c["name"]: c for c in self.data["crops"]}
        
        # Calculate basic allocation strategy from JSON data
        total_population = sum(d["population_x10k"] for d in self.data["irrigation_districts"])
        
        # Use land bounds from model parameters to get realistic area ranges
        initialized_count = 0
        total_area_allocated = 0
        
        for district_code in model.I:
            district_info = districts_data[district_code]
            district_pop = district_info["population_x10k"]
            
            # Calculate district's share of total agricultural activity
            pop_share = district_pop / total_population
            
            # Get the land bounds for this district from the model
            land_max = value(model.Lmax[district_code])  # Max land available
            land_min = value(model.Lmin[district_code])  # Min land required
            
            # Start with minimum required land per district
            district_total_area = land_min
            
            # Distribute this area among crops based on their market prices (profitability proxy)
            crop_prices = {crop: crops_data[crop]["market_price_CNY_per_kg"] for crop in model.J}
            total_price_weight = sum(crop_prices.values())
            
            for crop_name in model.J:
                # Allocate area proportional to crop profitability
                price_share = crop_prices[crop_name] / total_price_weight
                
                # Base allocation: each crop gets its price-weighted share of minimum land
                area = district_total_area * price_share
                
                # Add small random variation to avoid identical allocations
                import random
                area *= random.uniform(0.9, 1.1)  # ±10% variation
                
                # Ensure minimum viable area
                area = max(20, area)  # At least 20 hectares per crop
                
                # Round to whole hectares
                area = round(area)
                
                # Set the variable value
                model.Area[district_code, crop_name].set_value(area)
                initialized_count += 1
                total_area_allocated += area
                
                if verbose:
                    print(f"   📍 Area[{district_code},{crop_name[:5]}] = {area:4.0f} ha")
        
        if verbose:
            print(f"✅ Initialized {initialized_count} area variables")
            print(f"📊 Total area allocated: {total_area_allocated:,.0f} ha")
            print(f"🌾 Average area per district-crop: {total_area_allocated/initialized_count:.0f} ha")
        
        # Initialize water variables based on crop water requirements from data
        if verbose:
            print("💧 Initializing water variables based on crop coefficients...")
        water_initialized = 0
        
        for i in model.I:
            for j in model.J:
                area_val = value(model.Area[i,j])
                
                # Use actual crop coefficient data from JSON to estimate water needs
                crop_info = crops_data[j]
                
                # Sum up monthly crop coefficients to get total seasonal requirement
                total_kc = sum(crop_info["crop_coefficients"].values())
                
                # Use water requirement parameter from model (wreq) 
                base_water_req = value(model.wreq[j])  # mm per ha
                
                # Estimate water need: area × base_requirement × crop_coefficient
                estimated_water_need = area_val * base_water_req * total_kc * 10  # Convert to m³
                
                # Split between surface and groundwater based on district efficiency from JSON
                district_info = districts_data[i]
                surface_eff = district_info["water_coeff"]["surface"]
                ground_eff = district_info["water_coeff"]["ground"]
                
                # Use efficiency to determine preference (higher efficiency = more usage)
                total_eff = surface_eff + ground_eff
                surface_ratio = surface_eff / total_eff
                ground_ratio = ground_eff / total_eff
                
                surface_water = estimated_water_need * surface_ratio
                ground_water = estimated_water_need * ground_ratio
                
                # Set values for all scenarios
                for s in model.S:
                    # Note: SW and GW are Expressions, not Vars, so we initialize the underlying Vars
                    # We need to initialize SW_t and GW_t (the monthly variables)
                    months = list(model.T)
                    monthly_surface = surface_water / len(months)  # Distribute evenly across months
                    monthly_ground = ground_water / len(months)
                    
                    for t in months:
                        model.SW_t[i,j,s,t].set_value(monthly_surface)
                        model.GW_t[i,j,s,t].set_value(monthly_ground)
                        water_initialized += 2
        
        if verbose:
            print(f"✅ Initialized {water_initialized} water allocation variables")
        
        # Initialize yield variables based on water allocation and production functions
        if verbose:
            print("🌾 Initializing yield variables based on production functions...")
        yield_initialized = 0
        
        for i in model.I:
            for j in model.J:
                area_val = value(model.Area[i,j])
                crop_info = crops_data[j]
                
                # Get production function parameters from JSON
                a_param = crop_info["water_production_function"]["a"]
                b_param = crop_info["water_production_function"]["b"]
                
                # Estimate total water use (sum across months and scenarios)
                total_water = 0
                for s in model.S:
                    for t in model.T:
                        surface_val = value(model.SW_t[i,j,s,t]) if hasattr(model.SW_t[i,j,s,t], 'value') else 0
                        ground_val = value(model.GW_t[i,j,s,t]) if hasattr(model.GW_t[i,j,s,t], 'value') else 0
                        total_water += surface_val + ground_val
                
                # Average water per scenario
                avg_water_per_scenario = total_water / len(model.S)
                
                # Apply production function: Yield = a * water + b * area
                for s in model.S:
                    estimated_yield = a_param * avg_water_per_scenario + b_param * area_val
                    estimated_yield = max(0, estimated_yield)  # Ensure non-negative yield
                    
                    model.Yield[i,j,s].set_value(estimated_yield)
                    yield_initialized += 1
        
        if verbose:
            print(f"✅ Initialized {yield_initialized} yield variables")
            print("🎯 Initialization complete - ready for neural network approximation")

    # -------------------- helper -------------------- #
    def extract_data(self) -> Dict[str, Any]:
        if self.data_file is not None:
            with open(self.data_file, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            # If the file is a full problems dict, extract the right key
            if "problems" in all_data and "FanZhangWEFNexusModel" in all_data["problems"]:
                return all_data["problems"]["FanZhangWEFNexusModel"]
            else:
                return all_data
        try:
            with open(Path("problems") / "problem_data.json", "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except FileNotFoundError:
            with open("problem_data.json", "r", encoding="utf-8") as f:
                all_data = json.load(f)
        return all_data["problems"]["FanZhangWEFNexusModel"]

    # Two lightweight hooks Neuro2SP will call
    def first_stage_vars(self, model) -> List[Any]:
        """Return a *flat* list of first-stage VarData objects."""
        # Extract first-stage variables from the given model (not stored ones)
        fs_vars = []
        for i in model.I:
            for j in model.J:
                fs_vars.append(model.Area[i,j])
        return fs_vars
    
    def sample_scenario(self) -> Dict[str,Any]:
        """Draw a ξ from the underlying distribution (or pool)."""
        import random
        
        # Get scenario data with probabilities
        scenarios = self.data["hydrological_years"]
        names = [s["name"] for s in scenarios]
        probs = [s["probability"] for s in scenarios]
        
        # Sample one scenario based on probabilities
        chosen_scenario = random.choices(names, weights=probs, k=1)[0]
        
        return {"scenario": chosen_scenario}
