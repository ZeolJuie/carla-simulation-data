# sensors/lidar.py

import os
import time
import numpy as np
import open3d as o3d
import carla

import config
from sensors.sensor import Sensor

class LidarSensor(Sensor):
    def __init__(self, world, blueprint_library, walker, data_dir):
        super().__init__(world, blueprint_library, walker, data_dir, 'lidar')

    def _setup_sensor(self, blueprint_library, walker):
        # 设置激光雷达
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
        
        lidar_bp.set_attribute("upper_fov", str(config.LIDAR_UPPER_FOV))
        lidar_bp.set_attribute("lower_fov", str(config.LIDAR_LOWER_FOV))
        lidar_bp.set_attribute("channels", str(config.LIDAR_CHANNELS))
        lidar_bp.set_attribute("range", str(config.LIDAR_RANGE))
        lidar_bp.set_attribute("rotation_frequency", str(config.LIDAR_ROTATION_FREQUENCY))
        lidar_bp.set_attribute("points_per_second", str(config.LIDAR_POINTS_PER_SECOND))

        lidar_transform = carla.Transform(carla.Location(x=config.LIDAR_TRANSFORM_X, z=config.LIDAR_TRANSFORM_Z))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=walker)
        return lidar_bp, lidar
    
    def _save_data(self, sensor_data):
        """
        保存 LiDAR 点云到磁盘。
        """
        data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype("f4")))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        data[:, 1] = -data[:, 1]

        # 添加第五维ring index（全部设为0）
        # ring_index = np.zeros((data.shape[0], 1))
        # data = np.hstack((data, ring_index))

        file_path = os.path.join(f"{self.data_dir}/velodyne", '%06d.bin' % sensor_data.frame)
        data.astype(np.float32).tofile(file_path)

        print(f"Saved LiDAR point cloud to {file_path}")
