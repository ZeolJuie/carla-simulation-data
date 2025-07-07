import carla

from scenario.scenario import Scenario

scenario_01 = Scenario(
    sequence_id = '01',
    map_name = "Town10HD",
    start_point = carla.Location(x=-125.696838, y=10.016422, z=0.158620),
    end_point = carla.Location(x=-124.179443, y=80.482536, z=0.158620),
    description = ""
)

scenario_02 = Scenario(
    sequence_id = '02',
    map_name = "Town10HD",
    start_point = carla.Location(x=117, y=-7.054474, z=0.158620),
    end_point = carla.Location(x=89.022354, y=-4.088390, z=0.158620),
    description = ""
)

scenario_03 = Scenario(
    sequence_id = '03',
    map_name = "Town10HD",
    start_point = carla.Location(x=27.677645, y=58.019924, z=0.158620),
    end_point = carla.Location(x=91.465294, y=81.790596, z=0.158620),
    description = ""
)

scenario_04 = Scenario(
    sequence_id = '04',
    map_name = "Town10HD",
    start_point = carla.Location(x=-22.160627, y=121.606865, z=0.160169),
    end_point = carla. Location(x=47.558632, y=121.245651, z=0.158620),
    description = ""
)

scenario_05 = Scenario(
    sequence_id = '05',
    map_name = "Town10HD",
    start_point = carla.Location(x=-31.275158, y=56.675388, z=1.58620),
    end_point = carla.Location(x=27.677645, y=58.019924, z=0.158620),
    description = "Manual controll, walk through residential areas, complex obstacles"
)

scenario_06 = Scenario(
    sequence_id = '06',
    map_name = "Town01",
    start_point = carla.Location(x=264.260254, y=202.854172, z=0.105408),
    end_point = carla.Location(x=193.352036, y=204.662750, z=0.110720),
    description = ""
)

def generate_scenario(scenario_id: str):
    scenario_mapping = {
        '01': scenario_01,
        '02': scenario_02,
        '03': scenario_03,
        '04': scenario_04,
        '05': scenario_05,
        '06': scenario_06,
    }
    
    scenario = scenario_mapping.get(scenario_id)
    
    if scenario is None:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")
    
    return scenario
