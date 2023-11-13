from env_loop import test_and_report
from time import gmtime, strftime
from env.model_variable import ModelVariable
from util.functions import k_to_c, c_to_k, j_to_kwh, min_to_s, times_kilo, through_kilo, kwh_to_j, sin_encode_hour, cos_encode_hour, sin_encode_day, cos_encode_day

from env.house2 import House2Env

from util.log_helpers import setup_logging_directory

from logging import DEBUG, INFO
import util.logger as logger

# Environment parameters
ENV_NAME = "House2"
ALGORITHM = "StdCtr"
FMU_PATH = "./res/fmu/House2.fmu"
PRICES_PATH = "./res/data/house2.csv"
OUT_DIR = "./out"
TIME_STEP = 900 # [s]
NCP = 15
R_MAX = 3
NORMALIZE_OBS = False
FMU_LOG_LEVEL = 7

PRICE_CONFIG = {
    'dr':False,
    'p_el_purchase': 0.35,
    'p_el_feedin': 0.10,
    'p_gas_purchase': 0.10,
}

PARAMETER_VARS = [
    ModelVariable(name="InitUnixTime", value=1640995200, unit="s", info="Test for February 2022"),
    ModelVariable(name="MinimumControlEMS", value=True, unit="bool", info="Inputs are overwritten, if the EMS fails to keep the temperatures above the boundries of the standard control."),
    ModelVariable(name="dt_ems", value=15, unit="s", conv_calc=min_to_s, info="Timestep size EMS"),
    ModelVariable(name="TRefHeating", value=21, unit="K", conv_calc=c_to_k, info="Reference indoor temperature for heating"),
    ModelVariable(name="TRefCooling", value=24, unit="K", conv_calc=c_to_k, info="Reference indoor temperature for cooling"),
    ModelVariable(name="Tnight", value=21, unit="K", conv_calc=c_to_k, info="T_roomNightSet"),
    ModelVariable(name="NightTimeReductionStart_h", value=23, unit="h"),
    ModelVariable(name="NightTimeReductionEnd_h", value=6, unit="h"),
    ModelVariable(name="ActivateCooling", value=False, unit="bool", info="If true, cooling system is activated"),
    ModelVariable(name="nPeople", value=6),
    ModelVariable(name="HeatedArea", value=300, unit="m^2"),
    ModelVariable(name="YearlyElecConsumption", value=7000, unit="kWh"),
    ModelVariable(name="VStorage", value=785, unit="m^3", conv_calc=through_kilo, info="Volume of the thermal storage"),
    ModelVariable(name="UseBat", value=False, unit="bool", info="Use battery"),
    ModelVariable(name="PMaxBat", value=10, unit="W", conv_calc=times_kilo, info="Maximum charge/discharge power"),
    ModelVariable(name="EMaxBat", value=10, unit="J", conv_calc=kwh_to_j, info="Nominal maximum energy content"),
    ModelVariable(name="UsePV", value=True, unit="bool"),
    ModelVariable(name="PVPeak", value=10, unit="W", conv_calc=times_kilo, info="Installed peak power of the PV system"),
    ModelVariable(name="UseHP", value=True, unit="bool", info="Use heat pump"),
    ModelVariable(name="PHeatNom", value=5.75, unit="W", conv_calc=times_kilo, info="Nominal heat output at A2/W35 (not maximum heat output)"),
    ModelVariable(name="PColdNom", value=6.01, unit="W", conv_calc=times_kilo, info="Nominal cooling power at A35/W18 (not maximum cooling output)"),
    ModelVariable(name="PAuxMax", value=9, unit="W", conv_calc=times_kilo, info="Maximum power of the auxiliary heater"),
    ModelVariable(name="COPNom", value=4.65, info="Nominal COP at A2/W35"),
    ModelVariable(name="EERNom", value=5.92, info="Nominal EER at A35/W18")
]

OBSERVATION_VARS = [
    ModelVariable(name="HS_S_TM_VL_bM", unit="°C", info="Heat Sink flow temperature before mixing unit", conv_calc=k_to_c, bounds=[30.0, 80.0]),
    ModelVariable(name="HS_S_TM_HW_VL", unit="°C", info="Heat Sink flow temperature hot water", conv_calc=k_to_c, bounds=[30.0, 80.0]),
    ModelVariable(name="TS_S_TM_BT_9", unit="°C", info="Thermal storage temperature point 9", conv_calc=k_to_c, bounds=[30.0, 80.0]),
    ModelVariable(name="TS_S_TM_BT_4", unit="°C", info="Thermal storage temperature point 4", conv_calc=k_to_c, bounds=[30.0, 80.0]),
    ModelVariable(name="E_el_purchase", unit="kWh", info="Total energy that was purchased", conv_calc=j_to_kwh, bounds=[.0, 5.0], take_delta=True),
    ModelVariable(name="E_el_feedin", unit="kWh", info="Total energy that was fed in", conv_calc=j_to_kwh, bounds=[-1.0, .0], take_delta=True),
    ModelVariable(name="HP_E_elec", unit="kWh", info="Consumed electric energy of heat pump", conv_calc=j_to_kwh, bounds=[-4.0, .0], take_delta=True),
    ModelVariable(name="PV_E", unit="kWh", info="Heat pump flow temperature (output of the outdoor unit)", conv_calc=j_to_kwh,  bounds=[.0, 1.0], take_delta=True),
    ModelVariable(name="HP_S_TM_VL", unit="°C", info="heat pump flow temperature", conv_calc=k_to_c, bounds=[20.0, 80.0]),
    ModelVariable(name="TAmbient", unit="°C", info="Ambient temperature", conv_calc=k_to_c, bounds=[-20.0, 40.0]),
    ModelVariable(name="HourOfDay", unit="sin", info="Hour of day", conv_calc=sin_encode_hour),
    ModelVariable(name="HourOfDay", unit="cos", info="Hour of day", conv_calc=cos_encode_hour),
    ModelVariable(name="DayOfWeek", unit="sin", info="Day of week", conv_calc=sin_encode_day),
    ModelVariable(name="DayOfWeek", unit="cos", info="Day of week", conv_calc=cos_encode_day),
    # ModelVariable(name="Price_elBuy", unit="EUR/kWh", info="Electricity price buy", type="data_file", bounds=[-.11, .55]),
    # ModelVariable(name="Price_elSell", unit="EUR/kWh", info="Electricity price sell", type="data_file", bounds=[-.36, .3])
]

INFORMATIVE_VARS = [
    ModelVariable(name="HS_S_TM_Room", unit="°C", info="Building room temperature", conv_calc=k_to_c, bounds=[10.0, 40.0]), # t_room
    ModelVariable(name="HS_S_V_HW", unit="l", info="Total domestic hot water draw", conv_calc=times_kilo), # t_dhw
    ModelVariable(name="HS_E_DemElec", unit="kWh", info="Total electric energy demand", conv_calc=j_to_kwh),
    ModelVariable(name="HS_E_DemHeatHC", unit="kWh", info="Total heating energy demand", conv_calc=j_to_kwh),
    ModelVariable(name="HP_COP", info="Heat pump coefficiency of performance"),
    ModelVariable(name="TS_SOC_BT", info="State of charge of the thermal storage"),
    ModelVariable(name="Setpoint_Act_HP", info="Actual Setpoint"),
    ModelVariable(name="Setpoint_Act_HP_3WV", info="Actual Setpoint"),
    ModelVariable(name="Setpoint_Act_HP_Aux", info="Actual Setpoint"),
]

TEST_ENV_LENGTH = 31 * 24 * 60 * 60 # [s]

# Logging parameters
LOG_LEVEL = INFO
LOG_STD_INTERVAL = 500
LOG_TB_INTERVAL = 50


###################### END CONFIG ######################

# Init logging
run_id = strftime("%Y%m%d_%H%M%S", gmtime())
log_dir = setup_logging_directory(OUT_DIR, ENV_NAME, ALGORITHM, run_id)
log_file_name = f"{ENV_NAME}_{ALGORITHM.replace('/','_')}"
logger.root.setup(
    log_dir=log_dir,
    log_file=log_file_name,
    log_level=LOG_LEVEL,
    log_std_interval=LOG_STD_INTERVAL,
    log_tb_interval=LOG_TB_INTERVAL)
logger.log_dir = log_dir


test_env = House2Env(
    fmu_path=FMU_PATH,
    prices_path=PRICES_PATH,
    time_step=TIME_STEP,
    ncp=NCP,
    env_length=TEST_ENV_LENGTH,
    r_max=R_MAX,
    normalize_obs=NORMALIZE_OBS,
    price_config=PRICE_CONFIG,
    parameter_vars=PARAMETER_VARS,
    observation_vars=OBSERVATION_VARS,
    informative_vars=INFORMATIVE_VARS)


test_and_report(test_env, runs=1, epoch=0, std_ctr=True)
