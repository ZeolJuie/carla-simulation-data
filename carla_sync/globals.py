
import numpy as np
 
# Static global variable dictionary
_globals = { # Sorry for this below table's catogories are not correct cause that I can't find the right mapping table in the carla's document.
    'LABEL_COLORS':  np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
    (55, 90, 80), # Other, why it always output a inappropriate index
]) / 255.0, # normalize each channel [0-1] since is what Open3D uses,

    'CATEGOTIES': {15: 'bus', 14: 'car', 16: 'truck', 18: 'motorcycle', 19: 'bicycle', 13: 'Pedestrian'},
    'CATEGOTIES_INDEX': {'bus': 15, 'car': 14, 'truck': 16, 'motorcycle': 18,'bicycle': 19, 'Pedestrian': 13},
    
    # transform matrix, STANDARD means NUSCENES    
    'CAMERA_STANDARD_TO_CARLA': np.array([[0,  0,  1,  0],
                                     [1,  0,  0,  0],
                                     [0, -1,  0,  0],
                                     [0,  0,  0,  1]]),
    'EGO_CARLA_TO_STANDARD': np.array([[1,  0,  0,  0],
                                  [0, -1,  0,  0],
                                  [0,  0,  1,  0],
                                  [0,  0,  0,  1]]),
    'LIDAR_CARLA_TO_STANDARD': np.array([[0,  1,  0,  0],
                                    [1,  0,  0,  0],
                                    [0,  0,  1,  0],
                                    [0,  0,  0,  1]]),
    'WORLD_CARLA_TO_STANDARD': np.array([[1,  0,  0,  0],
                                   [0, -1,  0,  0],
                                   [0,  0,  1,  0],
                                   [0,  0,  0,  1]])
     
}

 
def get_global(name):
    return _globals[name]
 
def set_global(name, value): # add global variable only
    assert name not in _globals.keys()
    _globals[name] = value
 