import carla
import random

class Scenario:
    def __init__(self, sequence_id, map_name, start_point, end_point, ):
        # 基础配置
        self.sequence_id = sequence_id
        
        self.map_name = map_name
        self.start_point = start_point
        self.end_points = end_point

        
        
