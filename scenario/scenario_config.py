import carla

from scenario.scenario import Scenario


scenarios = [
    Scenario(
        sequence_id = '01',
        map_name = "Town10HD",
        start_point = carla.Location(x=-125.696838, y=10.016422, z=0.158620),
        end_point = carla.Location(x=-124.179443, y=80.482536, z=0.158620),
        description = ""
    ),

    Scenario(
        sequence_id = '02',
        map_name = "Town10HD",
        start_point = carla.Location(x=117, y=-7.054474, z=0.158620),
        end_point = carla.Location(x=89.022354, y=-4.088390, z=0.158620),
        description = ""
    ),

    Scenario(
        sequence_id = '03',
        map_name = "Town10HD",
        start_point = carla.Location(x=27.677645, y=58.019924, z=0.158620),
        end_point = carla.Location(x=91.465294, y=81.790596, z=0.158620),
        description = ""
    ),

    Scenario(
        sequence_id = '04',
        map_name = "Town10HD",
        start_point = carla.Location(x=-22.160627, y=121.606865, z=0.160169),
        end_point = carla. Location(x=47.558632, y=121.245651, z=0.158620),
        description = ""
    ),

    Scenario(
        sequence_id = '05',
        map_name = "Town10HD",
        start_point = carla.Location(x=-31.275158, y=56.675388, z=1.58620),
        end_point = carla.Location(x=27.677645, y=58.019924, z=0.158620),
        description = "Manual controll, walk through residential areas, complex obstacles"
    ),

    Scenario(
        sequence_id = '06',
        map_name = "Town10HD",
        start_point = carla.Location(x=114.122879, y=-36.800842, z=0.158620),
        end_point = carla.Location(x=62.706341, y=-75.494873, z=0.158620),
        description = ""
    ),

    Scenario(
        sequence_id = '07',
        map_name = "Town01",
        start_point = carla.Location(x=84.410812, y=210.617783, z=0.105408),
        end_point = carla.Location(x=83.739311, y=125.949089, z=0.105408),
        description = "Pedestrians go straight and cross the bridge"
    ),

    Scenario(
        sequence_id = '08',
        map_name = "Town01",
        start_point = carla.Location(x=123.108704, y=136.192139, z=0.105408),
        end_point = carla.Location(x=121.626068, y=191.188446, z=0.105408),
        description = "Pedestrians go straight and cross the bridge, double left-turn"
    ),

    Scenario(
        sequence_id = '09',
        map_name = "Town01",
        start_point = carla.Location(x=196.352036, y=200.662750, z=1.230720),
        end_point = carla.Location(x=252.039078, y=201.616547, z=0.107688),
        weather = carla.WeatherParameters.ClearNoon,
        description = "Manual controll"
    ),

    Scenario(
        sequence_id = '10',
        map_name = "Town01",
        start_point = carla.Location(x=30.287848, y=323.019562, z=0.109136),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "NOTE: Chairs and tables oot detected in (Semantic) LiDAR"
    ),

    Scenario(
        sequence_id = '30',
        map_name = "Town03",
        start_point = carla.Location(x=3.287439, y=-214.211899, z=0.160252),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "Long sequence capture and for clip segmentation"
    ),

    Scenario(
        sequence_id = '50',
        map_name = "Town04",
        start_point = carla.Location(x=56.713779, y=-233.425293, z=0.184139),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "Long sequence capture and for clip segmentation"
    ),

    Scenario(
        sequence_id = '60',
        map_name = "Town04",
        start_point = carla.Location(x=200.03, y=-238.78, z=0.69),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "Long sequence capture and for clip segmentation"
    ),

    Scenario(
        sequence_id = '70',
        map_name = "Town05",
        start_point = carla.Location(x=11.359136, y=81.200912, z=0.156784),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "Long sequence capture and for clip segmentation"
    ),

    Scenario(
        sequence_id = '80',
        map_name = "Town05",
        start_point = carla.Location(x=-115.59, y=-21.96, z=1.1),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "Long sequence capture and for clip segmentation"
    ),


    Scenario(
        sequence_id = '90',
        map_name = "Town10HD",
        start_point = carla.Location(x=-100.59, y=-21.96, z=1.1),
        end_point = carla.Location(x=27.677645, y=58.019924, z=0.158620),
        description = "Long sequence capture and for clip segmentation"
    ),


    
    
]



def generate_scenario(scenario_id: str):
    scenario_mapping = {s.sequence_id: s for s in scenarios}
    
    scenario = scenario_mapping.get(scenario_id)
    
    if scenario is None:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")
    
    return scenario
