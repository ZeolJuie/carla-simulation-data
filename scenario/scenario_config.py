import carla

from scenario.scenario import Scenario

scenario_01 = Scenario(
    sequence_id = '01',
    map_name = "Town10",
    start_point = carla.Location(x=-122.696838, y=26.016422, z=0.158620),
    end_point = carla.Location(x=-124.179443, y=80.482536, z=0.158620)
)

scenario_04 = Scenario(
    sequence_id = '04',
    map_name = "Town01",
    start_point = carla.Location(x=200, y=190, z=0.105408),
    end_point = carla.Location(x=121.626068, y=191.188446, z=0.105408)
)


def generate_scenario(scenario_id: str):
    scenario_mapping = {
        '01': scenario_01,
        '04': scenario_04
    }
    
    scenario = scenario_mapping.get(scenario_id)
    
    if scenario is None:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")
    
    return scenario
