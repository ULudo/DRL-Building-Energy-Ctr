import pytest

from src.env.simulation import SimulationResult, TUMLabSimulation


def test_TUMSimulation_simulate():

    tum_sim = TUMLabSimulation(
        url = 'opc.tcp://localhost:49580'
    )

    sim_result = tum_sim.simulate(0, 5)

    assert isinstance(sim_result.final("HP_S_TM_VL"), float)
    assert isinstance(sim_result.final("TAmbient"), float)
    assert isinstance(sim_result.final("E_el_purchase"), float)
    assert isinstance(sim_result.final("E_el_feedin"), float)
    assert isinstance(sim_result.final("HP_E_elec"), float)
    assert isinstance(sim_result.final("PV_E"), float)
    assert isinstance(sim_result.final("HS_S_TM_VL_bM"), float)
    assert isinstance(sim_result.final("HS_S_TM_HW_VL"), float)
    assert isinstance(sim_result.final("TS_S_TM_BT_9"), float)
    assert isinstance(sim_result.final("TS_S_TM_BT_4"), float)

    assert (sim_result.final("HP_starts") == 1.)
    assert (sim_result.final("HP_runtime") == -1)
    assert (sim_result.final("HourOfDay") == sim_result.get_hour_of_day())
    assert (sim_result.final("DayOfWeek") == sim_result.get_day())
    assert (sim_result.final("blabla") == None)


def test_TUMSimulation_set():
    tum_sim = TUMLabSimulation(
        url = 'opc.tcp://localhost:49580'
    )

    tum_sim.set(["", "HPModulation", "HPAuxModulation", "HP3WV"], controls=[0, 0.8, 0, 1])

    assert 1 == tum_sim.client.get_node('ns=2;s=SF2.ASHP_ctrl.A_V3_TL').get_value()
    assert abs(2 - tum_sim.client.get_node('ns=2;s=SF2.ASHP_ctrl.ASHP_mode').get_value()) < 0.0000000000001
    assert abs(0.8 - tum_sim.client.get_node('ns=2;s=SF2.ASHP_ctrl.Modulation_ASHP_set').get_value())  < 0.000000000001
    
    tum_sim.set(["", "HPModulation", "HPAuxModulation", "HP3WV"], controls=[0, 1, 0.6, 0])

    assert 0 == tum_sim.client.get_node('ns=2;s=SF2.ASHP_ctrl.A_V3_TL').get_value()
    assert abs(3 - tum_sim.client.get_node('ns=2;s=SF2.ASHP_ctrl.ASHP_mode').get_value()) < 0.0000000000001
    assert abs(0.6 - tum_sim.client.get_node('ns=2;s=SF2.ASHP_ctrl.Modulation_ASHP_set').get_value())  < 0.000000000001


def test_SimulationResult():
    sim_result = SimulationResult()
    
    sim_result.ASHP_flow_temp = 1
    sim_result.ambient_temp = 2
    sim_result.total_energy_purchased = 3
    sim_result.total_energy_feedin = 4
    sim_result.consumed_energy_ASHP = 5
    sim_result.generated_energy_PV = 6
    sim_result.heat_sink_flow_temp_before_mixing = 7
    sim_result.heat_sink_flow_temp_hot_water = 8
    sim_result.thermal_storage_point_9 = 9
    sim_result.thermal_storage_point_4 = 10

    assert (sim_result.final("HP_S_TM_VL") == 1)
    assert (sim_result.final("TAmbient") == 2)
    assert (sim_result.final("E_el_purchase") == 3)
    assert (sim_result.final("E_el_feedin") == 4)
    assert (sim_result.final("HP_E_elec") == 5)
    assert (sim_result.final("PV_E") == 6)
    assert (sim_result.final("HS_S_TM_VL_bM") == 7)
    assert (sim_result.final("HS_S_TM_HW_VL") == 8)
    assert (sim_result.final("TS_S_TM_BT_9") == 9)
    assert (sim_result.final("TS_S_TM_BT_4") == 10)

    assert (sim_result.final("HP_starts") == 1.)
    assert (sim_result.final("HP_runtime") == -1)
    assert (sim_result.final("HourOfDay") == sim_result.get_hour_of_day())
    assert (sim_result.final("DayOfWeek") == sim_result.get_day())
    assert (sim_result.final("blabla") == None)


