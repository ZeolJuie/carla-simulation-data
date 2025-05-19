import os
import numpy as np
import carla

import config
from sensors.sensor import Sensor

class RadarSensor(Sensor):
    def __init__(self, world, blueprint_library, vehicle, data_dir):
        super().__init__(world, blueprint_library, vehicle, data_dir, 'radar')
        
    def _setup_sensor(self, blueprint_library, walker):
        """Set up the radar sensor with specified parameters"""
        radar_bp = blueprint_library.find('sensor.other.radar')
        
        # Configure radar parameters
        radar_bp.set_attribute("horizontal_fov", str(90))  # 90 degree field of view（default）
        radar_bp.set_attribute("vertical_fov", str(30))    # 30 degree vertical fov
        radar_bp.set_attribute("points_per_second", str(15000))
        radar_bp.set_attribute("range", str(30))           # 30 meter range
        
        # Mount radar on vehicle roof
        radar_transform = carla.Transform(carla.Location(x=config.SENSOR_TRANSFORM_X, z=config.SENSOR_TRANSFORM_Z))
        radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=walker)
        return radar_bp, radar
    
    def _save_data(self, sensor_data):
        """
        Save radar detection data to disk.
        Radar data contains:
        - Depth (distance)
        - Azimuth angle
        - Altitude angle
        - Velocity (relative radial velocity)
        """
        # Convert raw data to numpy array
        # [vel, azimuth, altitude, depth]
        data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        data = np.reshape(data, (len(sensor_data), 4))  
        
        # Save as numpy binary file
        file_path = os.path.join(f"{self.data_dir}/radar", '%06d.npy' % sensor_data.frame)
        np.save(file_path, data)
        
        print(f"Saved radar data to {file_path}")
        
