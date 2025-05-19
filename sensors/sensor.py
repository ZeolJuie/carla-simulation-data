# sensors/sensor.py

import os
import time
import queue
import carla

class Sensor:
    def __init__(self, world, blueprint_library, walker, data_dir, sensor_type):
        self.world = world
        self.data_dir = data_dir
        self.sensor_type = sensor_type
        self.sensor_bp, self.sensor = self._setup_sensor(blueprint_library, walker)


    def _setup_sensor(self, blueprint_library, walker):
        raise NotImplementedError("Subclasses must implement this method")

    def _save_data(self, sensor_data):
        raise NotImplementedError("Subclasses must implement this method")

    def sensor_callback(self, sensor_data, sensor_queue, sensor_name):
        # 子类实现保存数据的逻辑
        self._save_data(sensor_data)
        # 父类处理通用的队列逻辑
        sensor_queue.put((sensor_data.frame, sensor_name))

    def start(self, sensor_queue, sensor_name):
        self.sensor.listen(lambda data: self.sensor_callback(data, sensor_queue, sensor_name))

    def get_transform(self):
        return self.sensor.get_transform()

    def get_attribute(self, attribute):
        return self.sensor_bp.get_attribute(attribute)

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()