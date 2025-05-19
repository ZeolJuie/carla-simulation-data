# sensors/camera.py

import os
import carla

import config
from sensors.sensor import Sensor

import numpy as np


class SemanticCameraSensor(Sensor):
    def __init__(self, world, blueprint_library, walker, data_dir):
        super().__init__(world, blueprint_library, walker, data_dir, 'semantic segmentation camera')

    def _setup_sensor(self, blueprint_library, walker):
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_IMAGE_SIZE_X))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_IMAGE_SIZE_Y))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', str(config.CAMERA_SENSOR_TICK))
        
        camera_transform = carla.Transform(carla.Location(x=config.SENSOR_TRANSFORM_X, z=config.SENSOR_TRANSFORM_Z))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=walker)
        return camera_bp, camera

    def _save_data(self, sensor_data):

        # 保存 语义分割相机数据 到磁盘。
        file_path = os.path.join(f"{self.data_dir}/semantic", '%06d.png' % sensor_data.frame)
        sensor_data.convert(carla.ColorConverter.CityScapesPalette)

        sensor_data.save_to_disk(file_path)
        print(f"Saved semantic image to {file_path}")

