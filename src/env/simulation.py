import time
import datetime
from pyfmi import load_fmu
from opcua import Client

from src.util.functions import suppress_stdout


class SimulationResult():
    """Class to hold the tum-lab simulation results"""
    
    def __init__(self) -> None:
        self.ASHP_flow_temp = None
        self.ambient_temp = None

        self.total_energy_purchased = None
        self.total_energy_feedin = None
        self.consumed_energy_ASHP = None
        self.generated_energy_PV = None

        self.heat_sink_flow_temp_before_mixing = None
        self.heat_sink_flow_temp_hot_water = None

        self.thermal_storage_point_9 = None
        self.thermal_storage_point_4 = None


    def final(self, key: str):
        """returns a specific value of the simulation"""

        # TODO: 
        # Erstelle Schnittstelle für HP_starts und HP_runtime zum TUM-Labor

        self.dict_read_vars = {
                      'HP_S_TM_VL': self.ASHP_flow_temp,
                      'TAmbient': self.ambient_temp,
                      'E_el_purchase': self.total_energy_purchased,
                      'E_el_feedin': self.total_energy_feedin,
                      'HP_E_elec': self.consumed_energy_ASHP,
                      'PV_E': self.generated_energy_PV,
                      'HS_S_TM_VL_bM': self.heat_sink_flow_temp_before_mixing,
                      'HS_S_TM_HW_VL': self.heat_sink_flow_temp_hot_water,
                      'TS_S_TM_BT_9': self.thermal_storage_point_9,
                      'TS_S_TM_BT_4': self.thermal_storage_point_4,

                      'HP_starts': 1.,
                      'HP_runtime': -1,
                      'HourOfDay': self.get_hour_of_day(),
                      'DayOfWeek': self.get_day()
        }
        try:
            return self.dict_read_vars[key]
        except:
            return None

    def get_hour_of_day(self):
        """returns hour of day in decimal (e.g. 9.75)"""

        now = datetime.datetime.now()
        return now.hour + now.minute/60
    
    def get_day(self):
        """returns day in decimal (e.g. 2.5 for Wednesday 12:00)"""

        now = datetime.datetime.now()
        return now.weekday() + self.get_hour_of_day() / 24.


class Simulation():
    """interface between environment and the simulation"""

    def __init__(self) -> None:
        pass

    def simulate(self):
        raise NotImplementedError("This method must be implemented by a subclass.")
    
    def set(self):
        raise NotImplementedError("This method must be implemented by a subclass.")
    
    def step(self):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def initialize(self):
        raise NotImplementedError("This method must be implemented by a subclass.")


class FMUSimulation(Simulation):
    """Simulation is in fmu format"""

    def __init__(
            self,
            fmu_path:str,
            fmu_type:str = "CS",
            fmu_log_level:int = 2,
            ncp:int = 15
    ) -> None:
        super().__init__()
        self.fmu = load_fmu(fmu_path, kind=fmu_type, log_level=fmu_log_level)
        
        self.fmu_opts = self.fmu.simulate_options()
        self.fmu_opts['ncp'] = ncp
        self.fmu_opts['initialize'] = False

    def simulate(self, start_time, final_time):
        """starts fmu simulation"""
        with suppress_stdout():
            return self.fmu.simulate(start_time=start_time, final_time=final_time, options=self.fmu_opts)

    def set(self, keys, vals) -> None:
        """sets fmu params"""
        return self.fmu.set(keys, vals)

    def reset(self) -> None:
        """reset fmu"""
        return self.fmu.reset()

    def initialize(self) -> None:
        """initialize fmu"""
        return self.fmu.initialize()


class TUMLabSimulation(Simulation):
    """Simulation is the CoSES Lab at TUM. Interface to CoSES is LabView"""
    
    def __init__(
            self,
            url:str
    ) -> None:
        super().__init__()
        
        # connect to LabView opcua
        self.client = Client(url=url)
        self.client.connect()

        # modulation parameters
        self.a_3v_tl: bool | None = None             
        self.ashp_mode: float | None = None               
        self.modulation_ashp_set: float| None = None    
        
    def simulate(self, start_time, final_time) -> SimulationResult:
        """waits for final_time-start_time [seconds] and returns the current state of the CoSES-Lab"""

        TIME_FRACTION = 30
        waiting_time = final_time-start_time
        waited_time = 0
        print("Time to wait [seconds]: ", waiting_time)
        while waited_time < waiting_time-TIME_FRACTION:
            print("Remaining time to wait [seconds]: ", waiting_time-waited_time)
            time.sleep(TIME_FRACTION)
            waited_time+=TIME_FRACTION
        time.sleep(waiting_time-waited_time)

        return self.get_observation()
    
    def get_observation(self) -> SimulationResult:
        """gets the values of the CoSES-Lab and returns them in a SimulationResult object"""
        
        sim_result = SimulationResult()

        sim_result.ASHP_flow_temp = self.client.get_node('ns=2;s=SF2.ASHP_read.S_TM-VL').get_value()
        sim_result.ambient_temp = self.client.get_node('ns=2;s=SF2.ASHP_read.S_TM-AS_In').get_value()

        sim_result.total_energy_purchased = self.client.get_node('ns=2;s=SF2.Eel_read.E_el_purchase').get_value()
        sim_result.total_energy_feedin = self.client.get_node('ns=2;s=SF2.Eel_read.E_el_feedin').get_value()
        sim_result.consumed_energy_ASHP = self.client.get_node('ns=2;s=SF2.Eel_read.E_el_ASHP').get_value()
        sim_result.generated_energy_PV = self.client.get_node('ns=2;s=SF2.Eel_read.E_el_PV').get_value()

        sim_result.heat_sink_flow_temp_before_mixing = self.client.get_node('ns=2;s=SF2.HS_read.S_TM-HC_VL_bM').get_value()
        sim_result.heat_sink_flow_temp_hot_water = self.client.get_node('ns=2;s=SF2.HS_read.S_TM-HW_VL').get_value()

        sim_result.thermal_storage_point_9 = self.client.get_node('ns=2;s=SF2.TS_read.S_TM-BT_9').get_value()
        sim_result.thermal_storage_point_4 = self.client.get_node('ns=2;s=SF2.TS_read.S_TM-BT_4').get_value()

        return sim_result

    def reset(self) -> None:
        """CoSES Lab can not be reset. Do nothing"""
        pass

    def set(self, input_keys, controls) -> None:
        """Applies input params to the CoSES Lab"""

        # This function is only able to set
        # a_3v_tl (HP3WV), 
        # ashp_mode (HPAuxModulation, HPModulation), 
        # modulation_ashp_set (HPAuxModulation, HPModulation)
        if input_keys[3] != "HP3WV": return
        if input_keys[2] != "HPAuxModulation": return
        if input_keys[1] != "HPModulation": return

        hp_modulation = float(controls[1])
        hp_aux_modulation = float(controls[2])

        # check if control values are valid
        if controls[3] == 1: 
            self.a_3v_tl = True
        elif controls[3] == 0:
            self.a_3v_tl = False
        else:
            raise Exception("a_3v_tl (hp_ts_port) needs to be 0 or 1")
        if hp_aux_modulation < 0.0 or hp_aux_modulation > 1.000000000001:
            raise Exception("hp_aux_modulation must be float <= 1.0 and >= 0.0")
        if hp_modulation < 0.0 or hp_modulation > 1.00000000001:
            raise Exception("hp_modulation must be float <= 1.0 and >= 0.0")
        if abs(hp_aux_modulation) < 0.00000000001:
            hp_aux_modulation = 0.0
        if abs(hp_modulation) < 0.00000000001:
            hp_modulation = 0.0

        # set control value
        if hp_aux_modulation == 0.0:
            self.ashp_mode = 2.0
            self.modulation_ashp_set = hp_modulation
        # if hp_aux_modulation is > 0, hp_modulation is assumed to be 100%
        else:
            self.ashp_mode = 3.0
            self.modulation_ashp_set = hp_aux_modulation

        # send control values to TUM-LAB
        self.client.get_node('ns=2;s=SF2.ASHP_ctrl.A_V3_TL').set_value(self.a_3v_tl)
        self.client.get_node('ns=2;s=SF2.ASHP_ctrl.ASHP_mode').set_value(self.ashp_mode)
        self.client.get_node('ns=2;s=SF2.ASHP_ctrl.Modulation_ASHP_set').set_value(self.modulation_ashp_set)

    def initialize(self):
        """skip"""
        pass
        