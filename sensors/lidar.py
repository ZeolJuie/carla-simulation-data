# carla_walker/sensors/lidar.py

import os
import time
import numpy as np
import open3d as o3d
import carla

import config

class LidarSensor:
    def __init__(self, world, blueprint_library, walker, data_dir):
        self.world = world
        self.data_dir = data_dir
        self.lidar = self._setup_lidar(blueprint_library, walker)

    def _setup_lidar(self, blueprint_library, walker):
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute("upper_fov", str(config.LIDAR_UPPER_FOV))
        lidar_bp.set_attribute("lower_fov", str(config.LIDAR_LOWER_FOV))
        lidar_bp.set_attribute("channels", str(config.LIDAR_CHANNELS))
        lidar_bp.set_attribute("range", str(config.LIDAR_RANGE))
        lidar_bp.set_attribute("rotation_frequency", str(config.LIDAR_ROTATION_FREQUENCY))
        lidar_bp.set_attribute("points_per_second", str(config.LIDAR_POINTS_PER_SECOND))
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=walker)
        return lidar

    def lidar_callback(self, point_cloud, lidar_queue):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype("f4")))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        points = data[:, :-1]
        points[:, 1] = -points[:, 1]

        timestamp = int(time.time() * 1000)
        file_path = os.path.join(self.data_dir, f"lidar_{timestamp}.ply")
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(file_path, o3d_point_cloud)
        lidar_queue.put(point_cloud.frame)
        print(f"保存点云数据: {file_path} (点数: {len(points)})")

    def sensor_callback(self, sensor_data, sensor_queue, sensor_name):
        # Do stuff with the sensor_data data like save it to disk
        # Then you just need to add to the queue
        sensor_queue.put((sensor_data.frame, sensor_name))

    def start(self, sensor_queue, sensor_name):
        self.lidar.listen(lambda data: self.sensor_callback(data, sensor_queue, sensor_name))

    def destroy(self):
        if self.lidar:
            self.lidar.destroy()