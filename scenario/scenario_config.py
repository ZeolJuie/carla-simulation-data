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
        description = "go straight and cross the bridge"
    ),

    Scenario(
        sequence_id = '08',
        map_name = "Town01",
        start_point = carla.Location(x=123.108704, y=136.192139, z=0.105408),
        end_point = carla.Location(x=121.626068, y=191.188446, z=0.105408),
        description = "go straight and cross the bridge, double left-turn"
    ),

    Scenario(
        sequence_id = '09',
        map_name = "Town01",
        start_point = carla.Location(x=108.207832, y=51.613781, z=0.105408),
        end_point = carla.Location(x=17.950916, y=-6.312346, z=0.105408),
        weather = carla.WeatherParameters.ClearNoon,
        description = "Manual controll. A pedestrian walks through a small town, passes a busy minor intersection, goes by a bus stop, and finally arrives in front of a house."
    ),

    Scenario(
        sequence_id = '10',
        map_name = "Town01",
        start_point = carla.Location(x=30.287848, y=323.019562, z=0.109136),
        end_point = carla.Location(x=4.920612, y=241.709656, z=0.105408),
        description = "NOTE: Chairs and tables oot detected in (Semantic) LiDAR"
    ),
    
    # ------------------------ 住宅区街道 --------------------
    Scenario(
        sequence_id = '11',
        map_name = "Town02",
        start_point = carla.Location(x=198, y=275, z=1.109136),
        end_point = carla.Location(x=198, y=185, z=1.105408),
        description = ""
    ),

    Scenario(
        sequence_id = '11',
        map_name = "Town02",
        start_point = carla.Location(x=198, y=275, z=1.109136),
        end_point = carla.Location(x=198, y=185, z=1.105408),
        description = ""
    ),

    Scenario(
        sequence_id = '12',
        map_name = "Town02",
        start_point = carla.Location(x=198, y=295, z=1.109136),
        end_point = carla.Location(x=198, y=185, z=1.105408),
        description = "Manual controll"
    ),

    Scenario(
        sequence_id = '13',
        map_name = "Town02",
        start_point = carla.Location(x=78, y=200, z=1.109136),
        end_point = carla.Location(x=198, y=185, z=1.105408),
        description = "Manual controll, Urban, many pedestrians on the sidewalk"
    ),

    Scenario(
        sequence_id = '14',
        map_name = "Town02",
        start_point = carla.Location(x=22, y=100, z=1.109136),
        end_point = carla.Location(x=198, y=185, z=1.105408),
        description = "Manual controll, Urban, Turn Left"
    ),

    Scenario(
        sequence_id = '25',
        map_name = "Town06",
        start_point = carla.Location(x=64.265198, y=-32.307999, z=0.206403),
        end_point = carla.Location(x=198, y=185, z=1.105408),
        description = ""
    ),

    Scenario(
        sequence_id = '35',
        map_name = "Town06",
        start_point = carla.Location(x=14.797142, y=-30.197739, z=1.196110),
        end_point = carla.Location(x=16.164064, y=121.149925, z=1.176403),
        description = ""
    ),

    Scenario(
        sequence_id = '36',
        map_name = "Town06",
        start_point = carla.Location(x=16.164064, y=121.149925, z=1.176403),
        end_point = carla.Location(x=16.164064, y=121.149925, z=1.176403),
        description = ""
    ),

    Scenario(
        sequence_id = '45',
        map_name = "Town07",
        start_point = carla.Location(x=-22.712111, y=-72.468132, z=1.160000),
        end_point = carla.Location(x=16.164064, y=121.149925, z=1.176403),
        description = ""
    ),

    Scenario(
        sequence_id = '45',
        map_name = "Town12",
        start_point = carla.Location(x=-22.712111, y=-72.468132, z=1.160000),
        end_point = carla.Location(x=16.164064, y=121.149925, z=1.176403),
        description = ""
    ),
    
]



def generate_scenario(scenario_id: str):
    scenario_mapping = {s.sequence_id: s for s in scenarios}
    
    scenario = scenario_mapping.get(scenario_id)
    
    if scenario is None:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")
    
    return scenario
