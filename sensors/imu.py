# carla_walker/sensors/imu.py

import os
import time
import numpy as np
import carla

class IMUSensor:
    def __init__(self, world, blueprint_library, walker, data_dir):
        self.world = world
        self.data_dir = data_dir
        self.imu = self._setup_imu(blueprint_library, walker)

    def _setup_imu(self, blueprint_library, walker):
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=walker)
        return imu

    def imu_callback(self, imu_data, imu_queue):
        transform_matrix = np.array(imu_data.transform.get_matrix())
        timestamp = int(time.time() * 1000)
        transform_file = os.path.join(self.data_dir, f"transform_{timestamp}.npy")
        np.save(transform_file, transform_matrix)
        print(f"保存转换矩阵: {transform_file}")

    def sensor_callback(self, sensor_data, sensor_queue, sensor_name):
        # Do stuff with the sensor_data data like save it to disk
        # Then you just need to add to the queue
        sensor_queue.put((sensor_data.frame, sensor_name))

    def start(self, sensor_queue, sensor_name):
        self.imu.listen(lambda data: self.sensor_callback(data, sensor_queue, sensor_name))

    def destroy(self):
        if self.imu:
            self.imu.destroy()