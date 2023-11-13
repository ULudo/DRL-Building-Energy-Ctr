import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from gymnasium import Env, spaces
from pyfmi import load_fmu

from util.functions import *
from env.model_variable import ModelVariable

import util.logger as logger


class House2Env(Env):

    o_cb_s_tm_vl = ModelVariable(name="CB_S_TM_VL", unit="°C", info="Condensing boiler flow temperature", conv_calc=k_to_c, bounds=[30.0, 80.0])
    o_hp_s_tm_vl = ModelVariable(name="HP_S_TM_VL", unit="°C", info="Heat pump flow temperature", conv_calc=k_to_c, bounds=[20.0, 80.0])
    o_s_tm_hw_vl = ModelVariable(name="HS_S_TM_HW_VL", unit="°C", info="Heat Sink flow temperature hot water", conv_calc=k_to_c, bounds=[30.0, 80.0])
    o_hs_s_tm_vl_bm = ModelVariable(name="HS_S_TM_VL_bM", unit="°C", info="Heat Sink flow temperature before mixing unit", conv_calc=k_to_c, bounds=[30.0, 80.0])
    o_pv_e = ModelVariable(name="PV_E", unit="kW", info="Produced energy of the PV system", conv_calc=j_to_kwh, bounds=[.0, 2.0])
    o_e_el_purchase = ModelVariable(name="E_el_purchase", value=.0, unit="kW", info="Total energy that was purchased", conv_calc=j_to_kwh, bounds=[.0, 5.0])
    o_e_el_feedin = ModelVariable(name="E_el_feedin", value=.0, unit="kW", info="Total energy that was fed in", conv_calc=j_to_kwh, bounds=[-2.0, .0])
    o_hp_runtime = ModelVariable(name="HP_runtime", unit="h", info="Total heat pump runtime", conv_calc=s_to_hour)
    o_hp_starts = ModelVariable(name="HP_starts", info="Total number of heat pump starts")
    o_price_el_buy = ModelVariable(name="Price_elBuy", unit="EUR/kWh", info="Electricity price buy")
    o_price_el_sell = ModelVariable(name="Price_elSell", unit="EUR/kWh", info="Electricity price sell")

    action_vars = [
        # Self control
        ModelVariable("CBModulation", unit="%", info="Set Modulation Condensing Boiler"),
        ModelVariable("HPModulation", unit="%", info="Modulation of the heat pump"),
        ModelVariable("HPAuxModulation", unit="%", info="Modulation of the auxiliary heater of the heat pump - if > 0 then HPModulation is turned to max"),
        ModelVariable("HP3WV", unit="bool", info="Switches between the inlet of the heat pump to the thermal storage (true: upper inlet)"),
        ModelVariable("HPMode", unit="bool", info="Heating mode (true) or cooling mode (false)"),
        
        # Automatic control
        ModelVariable("StandardContolST", unit="bool", info="Standard control of the solar thermal pump (STpump will be ignored)"),
        ModelVariable("StandardControlHP", unit="bool", info="Standard control of the HP (HPModulation and HPAuxModulation will be ignored)"),
        ModelVariable("StandardControlCB", unit="bool", info="Standard control of the CB (CBPower will be ignored)"),
        ModelVariable("StandardControlHP3WV", unit="bool", info="Standard control for switching the heat pump port (SwitchHPTSPort will be ignored)"),
        ModelVariable("StandardControlBattery", unit="bool", info="Standard control of the Battery (BatPCharge will be ignored)"),
        ModelVariable("StandardControlTRoom", unit="bool", info="Standard control for the room temperature")
    ]
    
    std_control_action = [.0, .0, .0, 0, 1, False, True, False, True, False, True]
    
    def __init__(
        self,
        fmu_path:str,
        fmu_type:str = "CS",
        price_config:Dict = {},
        observation_vars:List[ModelVariable] = [],
        parameter_vars:List[ModelVariable] = [],
        informative_vars:List[ModelVariable] = [],
        time_step:int = 900,
        ncp:int = 15,
        env_length:int = 1440,
        r_max:int = 4,
        normalize_obs:bool = True,
        prices_path:str = "",
        fmu_log_level:int = 2
    ):
        """OpenAI Gym environment for the ProHMo House2 FMU.
        Args:
            * fmu_path: Path to the FMU file.
            * fmu_type: FMU type. Either "ME" or "CS".
            * price_config: Dictionary with energy prices: 'dr':bool, 'p_el_purchase':float, 'p_el_feedin':float, 'p_gas_purchase':float
            * observation_vars: List of ModelVariables to be observed.
            * parameter_vars: List of ModelVariables to be set as FMU parameters.
            * informative_vars: List of ModelVariables to be retrieved for monitoring but not used in the observation space.
            * time_step: Simulation step size in seconds
            * ncp: Number of communication points per step.
            * env_length: Epoch length of the environment in seconds.
            * r_max: Maximum reward value.
            * normalize_obs: If the state features should be normalized using the bounds.
            * prices_path: Path to the CSV file with the electricity prices.
            * fmu_log_level: FMU log level. Default 2 (log error messages), 7 (log everything).
        """
        
        super(House2Env, self).__init__()
        
        self.time_step = time_step
        self.env_length = env_length
        self.r_max = r_max
        self.normalize_obs = normalize_obs
        self.observation_vars = observation_vars
        self.fmu_param_keys = [v.name for v in parameter_vars]
        self.fmu_param_vals = [v.conv_calc(v.value) for v in parameter_vars]
        self.fmu_input_keys = [v.name for v in self.action_vars]
        self.informative_vars = informative_vars
        
        # kW calculation for energy variables
        self.j_to_kw = JToKw(time_step/60)
        
        # Get the initial simulation time of the house
        var_init_time = [v for v in parameter_vars if v.name == "InitUnixTime"]
        assert var_init_time, f"No object found in parameter_vars with key [InitUnixTime]."
        self.init_time = var_init_time[0].value
        
        # Electricity prices based on a demand response program
        self.is_dr = price_config["dr"]
        if self.is_dr:
            self.df_prices = pd.read_csv(prices_path)
            self.df_prices[self.o_price_el_buy.name] = self.df_prices['Price'] + price_config['p_el_purchase']
            self.df_prices[self.o_price_el_sell.name] = self.df_prices['Price'] + price_config['p_el_feedin']
        else:
            self.prices = {
                self.o_price_el_buy.name: price_config['p_el_purchase'],
                self.o_price_el_sell.name: price_config['p_el_feedin']
            }
        
        # Create FMU
        self.fmu = load_fmu(fmu_path, kind=fmu_type, log_level=fmu_log_level)
        
        # Simulation options for FMU
        self.fmu_opts = self.fmu.simulate_options()
        self.fmu_opts['ncp'] = ncp
        self.fmu_opts['initialize'] = False
        
        # Gym variables.
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # Variable buffer for calculating the value difference between the previous and the current environment step.
        self.var_buffer = {v.name:v.value for v in observation_vars if v.take_delta}
        # Separate variable buffer for private variables to be independent from the variables in the observation space.
        self._var_buffer = {
            self.o_e_el_purchase.name: .0,
            self.o_e_el_feedin.name: .0
        }
    
    
    def _get_action_space(self) -> spaces.Box:
        """Defines the available actions of the environment."""
        raise NotImplementedError("This method must be implemented by a subclass.")
    
    
    def _get_observation_space(self) -> spaces.Box:
        """Defines the observation space according to the number of observation variables specified.
        """
        if self.normalize_obs:
            return spaces.Box(low=-1, high=1, dtype=np.float32, shape=(len(self.observation_vars),))
        else:
            return spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(len(self.observation_vars),))
    
    
    def _get_converted_value_from(self, fmu_key:ModelVariable) -> Union[float, int]:
        """Returns for the given ModelVariable the converted value of the current step.
        Args -- fmu_key: ModelVariable to be read.
        """
        return fmu_key.conv_calc(self.sim_result.final(fmu_key.name))
    
    
    def _get_current_e_price(self, key:str) -> float:
        """Retrieves the current electricity price based on the Unix timestamp from the DF if DR.
        Or returns the constant price value otherwise.
        Args -- key: Either o_price_el_buy.name or o_price_el_sell.name
        """
        if self.is_dr:
            unix_time = self.init_time + self.start
            return self.df_prices[(self.df_prices["unixtime"] <= unix_time)].iloc[-1][key]
        else:
            return self.prices[key]
    
    
    def _get_current_obs(self) -> np.ndarray:
        """Extract and convert observations from the simulation results.
        """
        obs = []
        for feature in self.observation_vars:
            val = self.sim_result.final(feature.name) if feature.type == "fmu" \
                else self._get_current_e_price(feature.name)
            if feature.name in self.var_buffer:
                val -= self.var_buffer[feature.name]
            val = feature.conv_calc(val)
            val = feature.normalize(val) if self.normalize_obs else val
            obs.append(val)
        return np.asarray(obs, dtype=np.float32)
    
    
    def _is_done(self) -> bool:
        """Checks if the HP temperature is above the maximum temperature 
        or if the simulation end time is reached.
        """
        # Prevent HP from overheating
        t_hp_max = self.o_hp_s_tm_vl.bounds[1]
        hp_vl_temp = self._get_converted_value_from(self.o_hp_s_tm_vl)
        logger.debug(f"Temperature at {self.o_hp_s_tm_vl.name}: {hp_vl_temp}")

        return (hp_vl_temp > t_hp_max) or (self.stop >= self.env_length)
    
    
    def _update_var_buffer(self):
        """Updates the variable buffer with the current simulation values.
        """
        for k in self.var_buffer: self.var_buffer[k] = self.sim_result.final(k)
    
    
    def _set_and_update_private_delta_vars(self):
        """Calculate the difference between the previous and current environment step 
        for privately used variables and update the private variable buffer.
        """
        self.o_e_el_purchase.value = self.sim_result.final(self.o_e_el_purchase.name) - self._var_buffer[self.o_e_el_purchase.name]
        self.o_e_el_feedin.value = self.sim_result.final(self.o_e_el_feedin.name) - self._var_buffer[self.o_e_el_feedin.name]
        self._var_buffer = {
            self.o_e_el_purchase.name: self.sim_result.final(self.o_e_el_purchase.name),
            self.o_e_el_feedin.name: self.sim_result.final(self.o_e_el_feedin.name)
        }
    
    
    def _simulate(self):
        """Execute environment simulation for the given time step.
        """
        logger.debug(f"Start simulating for [{self.start}, {self.stop}]")
        with suppress_stdout():
            self.sim_result = self.fmu.simulate(start_time=self.start, final_time=self.stop, options=self.fmu_opts)
        self.start = self.stop
        self.stop = self.start + self.time_step
        self.observation = self._get_current_obs()
        self._update_var_buffer()
        self.done = self._is_done()
        self._set_and_update_private_delta_vars()
    
    
    def _get_environment_report(self) -> Dict[str, Union[float, int]]:
        """Returns a summary of the energy resources consumed and produced by the building.
        """
        
        e_supply = self._get_converted_value_from(self.o_e_el_purchase)
        e_feed = self._get_converted_value_from(self.o_e_el_feedin)
        e_pv = self._get_converted_value_from(self.o_pv_e)
        hp_starts = self._get_converted_value_from(self.o_hp_starts)
        hp_operation_hours = self._get_converted_value_from(self.o_hp_runtime)
        
        report = {
            "Electricity consumption": -e_supply,
            "Electricity production": -e_feed,
            "Total electricity consumption": -(e_supply + e_feed),
            "PV production": e_pv,
            "Constraint violations": self.constraint_violations,
            "Total electricity costs": self.total_electricity_costs,
            "Number of starts HP": hp_starts,
            "Operation hours HP": hp_operation_hours
        }
        
        return report
    
    
    def get_observation_details(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """Returns the state values of the current environment step with a time resolution of ncp.
        """
        
        self.observation_space
        obs_details = {v.name:list(self.sim_result[v.name]) \
            for v in self.observation_vars + self.informative_vars if v.type == "fmu"}

        if self.is_dr:
            el_price_buy, el_price_sell = self._get_current_e_price(self.o_price_el_buy.name), self._get_current_e_price(self.o_price_el_sell.name) # [EUR/kWh]
            obs_details["ElectricityPriceBuy(EUR/kWh)"] = [el_price_buy for i in range(len(obs_details[list(obs_details.keys())[0]]))]
            obs_details["ElectricityPriceSell(EUR/kWh)"] = [el_price_sell for i in range(len(obs_details[list(obs_details.keys())[0]]))]
            
        return obs_details
    
    
    def _fmu_gim_step(self, controls:List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, any]]:
        """Runs the building for one time step with the given controls and
        returns the results in OpenAI Gym API format.
        Args -- controls: List of the control values for the FMU.
        """
        # Simulate with given actions.
        self.fmu.set(self.fmu_input_keys, controls)
        self._simulate()
        
        # Calculate reward of the current building state.
        self.reward, self.info = self._calculate_reward()
        self.info["action"] = controls
        
        # If done read consumed and produced energy resources.
        if self.done:
            logger.debug("Environment is done. Determining consumed and produced building energy resources.")
            self.info["final_resources"] = self._get_environment_report()
        
        return self.observation, self.reward, self.done, False, self.info
    
    
    def step(self, action:List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, any]]:
        """OpenAI Gym API step. Executes one time step of the environment.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")


    def standard_control_step(self) -> Tuple[List[float], float, bool, bool, Dict[str, any]]:
        """Runs the building for one time step with standard control.
        """
        return self._fmu_gim_step(self.std_control_action)
    

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,) -> Tuple[List[float], dict]:
        """OpenAI Gym API reset. Resets the building to the initial state and initializes the FMU.
        """
        super().reset(seed=seed)
        
        logger.debug("Experiment reset was called. Resetting the model.")

        # Reset environment values.
        for k in self.var_buffer: self.var_buffer[k] = .0
        self.constraint_violations = 0
        self.total_electricity_costs = 0
        
        # Reset fmu.
        self.fmu.reset()
        
        # Init environment.
        self.fmu.set(self.fmu_param_keys, self.fmu_param_vals)
        self.fmu.initialize()

        # Get initial observation
        self.start, self.stop = 0, 0
        self.fmu.set(self.fmu_input_keys, self.std_control_action)
        self._simulate()

        return self.observation, {}
    
    
    def _calculate_reward(self) -> Tuple[float, Dict[str, Union[float, int]]]:
        """Calculates the reward based on the current building state.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")
    

class House2HPand3WVEnv(House2Env):
    
    std_control_off = [False, False, False, False, False, True]
    
    
    def __init__(self,
     fmu_path: str,
     fmu_type: str = "CS",
     price_config: Dict = {},
     observation_vars: List[ModelVariable] = [],
     parameter_vars: List[ModelVariable] = [],
     informative_vars: List[ModelVariable] = [],
     time_step: int = 900,
     ncp: int = 15,
     env_length: int = 1440,
     r_max: int = 4,
     normalize_obs: bool = True,
     prices_path: str = "",
     fmu_log_level: int = 2):
    
        super().__init__(fmu_path,
            fmu_type,
            price_config,
            observation_vars,
            parameter_vars,
            informative_vars,
            time_step,
            ncp,
            env_length,
            r_max,
            normalize_obs,
            prices_path,
            fmu_log_level)
    
    
    def _get_action_space(self) -> spaces.Box:
        """Available actions of the environment:
            * HP modulation
            * HP aux. modulation
            * HP TS port
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    
    def step(self, action:List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, any]]:
        """OpenAI Gym API step. Executes one time step of the environment. Action values are
        converted to the appropriate FMU input values.
        Args -- action: Vector of 3 action values. Lie between [-1, 1].
        """
        logger.debug("Step function call.")
        
        # Condensing boiler is not used.
        cb_modulation = .0
        
        # Mapping for HP modulation.
        hp_modulation = ((action[0] + 1) / 2) * 0.9 + 0.1
        hp_modulation = 0 if hp_modulation < 0.2 else hp_modulation
        
        # Mapping for aux. modulation.
        hp_aux_modulation = action[1]
        if    hp_aux_modulation < -0.5: hp_aux_modulation = 0    # aux. at 0 %
        elif  hp_aux_modulation < 0   : hp_aux_modulation = 0.1  # aux. at 33 %
        elif  hp_aux_modulation < 0.5 : hp_aux_modulation = 0.34 # aux. at 67 %
        else: hp_aux_modulation = 0.68 # aux. at 100 %
        
        # Mapping for the TS port.
        hp_ts_port = 0 if action[2] < 0 else 1
        
        # Simulate with action array.
        actions = [cb_modulation, hp_modulation, hp_aux_modulation, hp_ts_port, 1, *self.std_control_off]
        return self._fmu_gim_step(actions)
    
    
    def _calculate_reward(self) -> Tuple[float, Dict[str, Union[float, int]]]:
        """Calculates the reward based on the current building state.
        The reward function depends on the electricity price and temperature constrains.
        """
        
        # Temperature values.
        t_hp = self._get_converted_value_from(self.o_hp_s_tm_vl)
        t_dhw = self._get_converted_value_from(self.o_s_tm_hw_vl)
        t_heating = self._get_converted_value_from(self.o_hs_s_tm_vl_bm)
        # Electricity consumption and production.
        el_supply = self.j_to_kw(self.o_e_el_purchase.value)
        el_feed = self.j_to_kw(self.o_e_el_feedin.value)
        
        # Primary goal: Keeping temperature constraints (heating and DHW).
        if t_heating >= 35 and t_dhw >= 55:
            rp_t = self.r_max
        else:
            rp_t = 0
            self.constraint_violations += 1
        
        # Secondary goal: Electricity price reduction.
        el_price_buy, el_price_sell = self._get_current_e_price(self.o_price_el_buy.name), self._get_current_e_price(self.o_price_el_sell.name) # [EUR/kWh]
        rs_el_buy, rs_el_sell = -el_supply * el_price_buy, -el_feed * el_price_sell # EUR
        self.total_electricity_costs += rs_el_buy + rs_el_sell
        
        rw_description = {
            "rw primary goal": rp_t,
            "electricity price buy in EUR/kWh": el_price_buy,
            "electricity price sell in EUR/kWh": el_price_sell,
            "electricity demand in kW": -el_supply,
            "electricity produced in kW": -el_feed,
            "rw electricity bought in EUR": rs_el_buy,
            "rw electricity sold in EUR": rs_el_sell
        }
        
        # Calculate total reward.
        rw = np.max([.0, rp_t + rs_el_buy + rs_el_sell])
        rw = -self.r_max if (t_hp > self.o_hp_s_tm_vl.bounds[1]) else rw
        
        return rw, rw_description


class House2HPEnv(House2Env):
    
    std_control_off = [False, False, False, True, False, True]
    
    
    def __init__(self,
     fmu_path: str,
     fmu_type: str = "CS",
     price_config: Dict = {},
     observation_vars: List[ModelVariable] = [],
     parameter_vars: List[ModelVariable] = [],
     informative_vars: List[ModelVariable] = [],
     time_step: int = 900,
     ncp: int = 15,
     env_length: int = 1440,
     r_max: int = 4,
     normalize_obs: bool = True,
     prices_path: str = "",
     fmu_log_level: int = 2):
    
        super().__init__(fmu_path,
            fmu_type,
            price_config,
            observation_vars,
            parameter_vars,
            informative_vars,
            time_step,
            ncp,
            env_length,
            r_max,
            normalize_obs,
            prices_path,
            fmu_log_level)
    
    
    def _get_action_space(self) -> spaces.Box:
        """Available actions of the environment:
            * HP modulation
            * HP aux. modulation
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    
    def step(self, action:List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, any]]:
        """OpenAI Gym API step. Executes one time step of the environment. Action values are
        converted to the appropriate FMU input values.
        Args -- action: Vector of 2 action values. Lie between [-1, 1].
        """
        logger.debug("Step function call.")
        
        # Condensing boiler is not used.
        cb_modulation = .0
        
        # Mapping for HP modulation.
        hp_modulation = ((action[0] + 1) / 2) * 0.9 + 0.1
        hp_modulation = 0 if hp_modulation < 0.2 else hp_modulation
        
        # Mapping for aux. modulation.
        hp_aux_modulation = action[1]
        if    hp_aux_modulation < -0.5: hp_aux_modulation = 0    # aux. at 0 %
        elif  hp_aux_modulation < 0   : hp_aux_modulation = 0.1  # aux. at 33 %
        elif  hp_aux_modulation < 0.5 : hp_aux_modulation = 0.34 # aux. at 67 %
        else: hp_aux_modulation = 0.68 # aux. at 100 %
        
        # Simulate with action array.
        actions = [cb_modulation, hp_modulation, hp_aux_modulation, 0, 1, *self.std_control_off]
        return self._fmu_gim_step(actions)
    
    
    def _calculate_reward(self) -> Tuple[float, Dict[str, Union[float, int]]]:
        """Calculates the reward based on the current building state.
        The reward function only depends on the electricity price. 
        """
        
        # Electricity consumption and production.
        el_supply = self.j_to_kw(self.o_e_el_purchase.value)
        el_feed = self.j_to_kw(self.o_e_el_feedin.value)
        
        # Electricity price reduction.
        el_price_buy, el_price_sell = self._get_current_e_price(self.o_price_el_buy.name), self._get_current_e_price(self.o_price_el_sell.name) # [EUR/kWh]
        rs_el_buy, rs_el_sell = -el_supply * el_price_buy, -el_feed * el_price_sell # EUR
        self.total_electricity_costs += rs_el_buy + rs_el_sell
        
        rw_description = {
            "electricity price buy in EUR/kWh": el_price_buy,
            "electricity price sell in EUR/kWh": el_price_sell,
            "electricity demand in kW": -el_supply,
            "electricity produced in kW": -el_feed,
            "rw electricity bought in EUR": rs_el_buy,
            "rw electricity sold in EUR": rs_el_sell
        }
        
        # Calculate total reward.
        rw = rs_el_buy + rs_el_sell
        
        return rw, rw_description