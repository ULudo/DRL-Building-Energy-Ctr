import os
import pickle
from time import gmtime, strftime
from env.model_variable import ModelVariable
from util.functions import k_to_c, c_to_k, j_to_kwh, min_to_s, times_kilo, through_kilo, kwh_to_j, sin_encode_hour, cos_encode_hour, sin_encode_day, cos_encode_day

from env.house2 import House2HPand3WVEnv
from agent.rsac import RecurrentSAC
from agent.recurrent_replay_buffer import RecurrentReplayBuffer

from env_loop import train
from util.log_helpers import setup_logging_directory
from util.functions import set_seeds

from logging import DEBUG, INFO
import util.logger as logger

##################### BEGIN CONFIG #####################

# Trains a recurrent SAC agent on the ProHMo House2. The agent controls
# the heat pump and the three-way valve of the thermal storage. The reward
# depends on the indoor temperature and the electricity price. While training
# the minimum control is turned off, while testing it is turned on.

# Environment parameters
ENV_NAME = "House2"
ALGORITHM = "RSAC"
FMU_PATH = "./res/fmu/House2.fmu"
PRICES_PATH = "./res/data/house2.csv"
OUT_DIR = "./out"
TIME_STEP = 900 # [s]
NCP = 15
R_MAX = 3
NORMALIZE_OBS = True
FMU_LOG_LEVEL = 7

PRICE_CONFIG = {
    'dr':True,
    'p_el_purchase': 0.20,
    'p_el_feedin': .0,
    'p_gas_purchase': 0.10,
}

PARAMETER_VARS = [
    ModelVariable(name="dt_ems", value=15, unit="min", conv_calc=min_to_s, info="Timestep size EMS"),
    ModelVariable(name="TRefHeating", value=21, unit="°C", conv_calc=c_to_k, info="Reference indoor temperature for heating"),
    ModelVariable(name="TRefCooling", value=24, unit="°C", conv_calc=c_to_k, info="Reference indoor temperature for cooling"),
    ModelVariable(name="Tnight", value=21, unit="°C", conv_calc=c_to_k, info="T_roomNightSet"),
    ModelVariable(name="NightTimeReductionStart_h", value=23, unit="h"),
    ModelVariable(name="NightTimeReductionEnd_h", value=6, unit="h"),
    ModelVariable(name="ActivateCooling", value=False, unit="bool", info="If true, cooling system is activated"),
    ModelVariable(name="nPeople", value=6),
    ModelVariable(name="HeatedArea", value=300, unit="m^2"),
    ModelVariable(name="YearlyElecConsumption", value=7000, unit="kWh"),
    ModelVariable(name="VStorage", value=785, unit="m^3", conv_calc=through_kilo, info="Volume of the thermal storage"),
    ModelVariable(name="UseBat", value=False, unit="bool", info="Use battery"),
    ModelVariable(name="PMaxBat", value=10, unit="kW", conv_calc=times_kilo, info="Maximum charge/discharge power"),
    ModelVariable(name="EMaxBat", value=10, unit="kWh", conv_calc=kwh_to_j, info="Nominal maximum energy content"),
    ModelVariable(name="UsePV", value=True, unit="bool"),
    ModelVariable(name="PVPeak", value=20, unit="kW", conv_calc=times_kilo, info="Installed peak power of the PV system"), # TODO
    ModelVariable(name="UseHP", value=True, unit="bool", info="Use heat pump"),
    ModelVariable(name="PHeatNom", value=5.75, unit="kW", conv_calc=times_kilo, info="Nominal heat output at A2/W35 (not maximum heat output)"),
    ModelVariable(name="PColdNom", value=6.01, unit="kW", conv_calc=times_kilo, info="Nominal cooling power at A35/W18 (not maximum cooling output)"),
    ModelVariable(name="PAuxMax", value=9, unit="kW", conv_calc=times_kilo, info="Maximum power of the auxiliary heater"),
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
    ModelVariable(name="Price_elBuy", unit="EUR/kWh", info="Electricity price buy", type="data_file"),
    ModelVariable(name="Price_elSell", unit="EUR/kWh", info="Electricity price sell", type="data_file"),
]

INFORMATIVE_VARS = [
    ModelVariable(name="HS_S_TM_Room", unit="°C", info="Building room temperature", conv_calc=k_to_c, bounds=[10.0, 40.0]),
    ModelVariable(name="HS_S_V_HW", unit="l", info="Total domestic hot water draw", conv_calc=times_kilo),
    ModelVariable(name="HS_E_DemElec", unit="kWh", info="Total electric energy demand", conv_calc=j_to_kwh),
    ModelVariable(name="HS_E_DemHeatHC", unit="kWh", info="Total heating energy demand", conv_calc=j_to_kwh),
    ModelVariable(name="HP_COP", info="Heat pump coefficiency of performance"),
    ModelVariable(name="TS_SOC_BT", info="State of charge of the thermal storage"),
    ModelVariable(name="Setpoint_Act_HP", info="Actual Setpoint"),
    ModelVariable(name="Setpoint_Act_HP_3WV", info="Actual Setpoint"),
    ModelVariable(name="Setpoint_Act_HP_Aux", info="Actual Setpoint"),
]

# Training environment specific parameters
TRAIN_SPECIFIC_PARAMETER_VARS = [
    ModelVariable(name="InitUnixTime", value=1609459200, unit="s", info="Train for January 2021"),
    ModelVariable(name="MinimumControlEMS", value=False, unit="bool", info="Inputs can't be overwritten, if the EMS fails to keep the temperatures above the boundries of the standard control."),
]
TRAIN_ENV_LENGTH = 31 * 24 * 60 * 60 # [s]

# Testing environment specific parameters
TEST_SPECIFIC_PARAMETER_VARS = [
    ModelVariable(name="InitUnixTime", value=1640995200, unit="s", info="Test for January 2022"),
    ModelVariable(name="MinimumControlEMS", value=True, unit="bool", info="Inputs are overwritten, if the EMS fails to keep the temperatures above the boundries of the standard control.")
]
TEST_ENV_LENGTH = 31 * 24 * 60 * 60 # [s]

# Model parameters
HIDDEN_LSTM_UNITS = 256

# Replay buffer parameters
REPLAY_BUFFER_CAPACITY = 100_000
SLIDING_WIN_LEN = 24
BATCH_SIZE = 64
BUFFER_SAVE_FILE = "./recurrent_replay_buffer.pkl"
BUFFER_LOAD_FILE = ""

# Training parameters
NUM_EPOCHS = 10
NUM_STEPS_PER_EPOCH = 10_000
NUM_TEST_EPISODES_PER_EPOCH = 1
UPDATE_AFTER = 512

# Logging parameters
LOG_LEVEL = INFO
LOG_STD_INTERVAL = 500
LOG_TB_INTERVAL = 50

# Seed
SEED = 42

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

# Create actor save directory
actor_save_dir = os.path.join(log_dir, "actor")
if not os.path.exists(actor_save_dir):
        os.makedirs(actor_save_dir)

# Setup environments
train_env = House2HPand3WVEnv(
    fmu_path=FMU_PATH,
    prices_path=PRICES_PATH,
    time_step=TIME_STEP,
    ncp=NCP,
    env_length=TRAIN_ENV_LENGTH,
    r_max=R_MAX,
    normalize_obs=NORMALIZE_OBS,
    price_config=PRICE_CONFIG,
    parameter_vars=PARAMETER_VARS + TRAIN_SPECIFIC_PARAMETER_VARS,
    observation_vars=OBSERVATION_VARS,
    informative_vars=INFORMATIVE_VARS,
    fmu_log_level=FMU_LOG_LEVEL)
obs_dim = train_env.observation_space.shape[0]
act_dim = train_env.action_space.shape[0]

test_env = House2HPand3WVEnv(
    fmu_path=FMU_PATH,
    prices_path=PRICES_PATH,
    time_step=TIME_STEP,
    ncp=NCP,
    env_length=TEST_ENV_LENGTH,
    r_max=R_MAX,
    normalize_obs=NORMALIZE_OBS,
    price_config=PRICE_CONFIG,
    parameter_vars=PARAMETER_VARS + TEST_SPECIFIC_PARAMETER_VARS,
    observation_vars=OBSERVATION_VARS,
    informative_vars=INFORMATIVE_VARS)

# Set seed if given
if SEED is not None:
    set_seeds(SEED)

# Init recurrent SAC algorithm
rsac_algorithm = RecurrentSAC(
    input_dim=obs_dim,
    action_dim=act_dim,
    hidden_dim=HIDDEN_LSTM_UNITS
)

# Init replay buffer
assert BUFFER_LOAD_FILE == '' or BUFFER_SAVE_FILE == '', "Cannot load and save buffer at the same time."
if BUFFER_LOAD_FILE:
    bl_dir = os.path.join(BUFFER_LOAD_FILE, "replay_buffer.pkl")
    with open(bl_dir, 'rb') as inp:
        replay_buffer = pickle.load(inp)
else:
    replay_buffer = RecurrentReplayBuffer(
        o_dim=obs_dim,
        a_dim=act_dim,
        sliding_win_len=SLIDING_WIN_LEN,
        capacity=REPLAY_BUFFER_CAPACITY,
        batch_size=BATCH_SIZE
    )

# Start training loop
train(
    env=train_env,
    algorithm=rsac_algorithm,
    buffer=replay_buffer,
    num_epochs=NUM_EPOCHS,
    num_steps_per_epoch=NUM_STEPS_PER_EPOCH,
    num_test_episodes_per_epoch=NUM_TEST_EPISODES_PER_EPOCH,
    update_after=UPDATE_AFTER,
    test_env=test_env,
    actor_save_dir=actor_save_dir,
    buffer_save_file=BUFFER_SAVE_FILE
)

