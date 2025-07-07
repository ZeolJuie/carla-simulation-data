# sensors/depth_camera.py

import os
import carla

import config
from sensors.sensor import Sensor

import numpy as np


class DepthCameraSensor(Sensor):
    def __init__(self, world, blueprint_library, walker, data_dir):
        super().__init__(world, blueprint_library, walker, data_dir, 'depth_camera')

    def _setup_sensor(self, blueprint_library, walker):
        camera_bp = blueprint_library.find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_IMAGE_SIZE_X))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_IMAGE_SIZE_Y))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', str(config.CAMERA_SENSOR_TICK))
        camera_transform = carla.Transform(carla.Location(x=config.SENSOR_TRANSFORM_X, z=config.SENSOR_TRANSFORM_Z))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=walker)
        return camera_bp, camera

    def _save_data(self, sensor_data):
        
        # 保存 深度相机数据 到磁盘。

        # 将深度编码的RGB图像还原到深度信息，并保存numpy数组，但是一帧对应的深度信息有15MB左右
        # file_path = os.path.join(f"{self.data_dir}/depth", '%06d.npy' % sensor_data.frame)
        # depth_image = self.convert_depth_data(sensor_data)
        # np.save(file_path, depth_image)

        file_path = os.path.join(f"{self.data_dir}/depth", '%06d.jpg' % sensor_data.frame)

        # 直接保存原始数据
        sensor_data.save_to_disk(file_path)
        
        # 将原始数据转换为深度图像并保存
        # sensor_data.save_to_disk(file_path, carla.ColorConverter.LogarithmicDepth)
        print(f"Saved Depth image to {file_path}")

    def convert_depth_data(self, sensor_data):
        """
        将 CARLA 的深度图像数据转换为实际的深度值（以米为单位）。
        """
        # 将原始数据转换为数组
        depth_array = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
        depth_array = np.reshape(depth_array, (sensor_data.height, sensor_data.width, 4))  # RGBA格式
        
        # 提取深度信息（R, G, B 通道组合成一个浮点数）
        depth_meters = np.dot(depth_array[:, :, :3], [65536.0, 256.0, 1.0]) / (256 * 256 * 256 - 1)
        depth_meters *= 1000  # 转换为米
        
        return depth_meters