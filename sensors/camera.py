# sensors/camera.py

import os
import carla

import config
from sensors.sensor import Sensor

class CameraSensor(Sensor):
    def __init__(self, world, blueprint_library, walker, data_dir, camera_config):
        """
        Args:
            world: CARLA世界对象
            blueprint_library: 蓝图库
            walker: 要绑定的行人actor
            data_dir: 数据存储目录
            camera_config: 相机配置字典，包含:
                - name
                - image_size_x: 图像宽度
                - image_size_y: 图像高度
                - fov: 视野角度
                - sensor_tick: 传感器采样间隔
                - transforms: 多个相机的变换配置列表
                    - location: 位置(x,y,z)
                    - rotation: 旋转(pitch,yaw,roll)
        """
        self.camera_config = camera_config
        super().__init__(world, blueprint_library, walker, data_dir, camera_config['name'])
        
    def _setup_sensor(self, blueprint_library, walker):
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_IMAGE_SIZE_X))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_IMAGE_SIZE_Y))
        camera_bp.set_attribute('fov', str(self.camera_config['fov']))
        camera_bp.set_attribute('sensor_tick', str(config.CAMERA_SENSOR_TICK))

        location = carla.Location(
            x=self.camera_config['transforms']['location']['x'],
            y=self.camera_config['transforms']['location']['y'],
            z=self.camera_config['transforms']['location']['z']
        )

        rotation = carla.Rotation(
            pitch=self.camera_config['transforms']['rotation']['pitch'],
            yaw=self.camera_config['transforms']['rotation']['yaw'],
            roll=self.camera_config['transforms']['rotation']['roll']
        )

        camera_transform = carla.Transform(location, rotation)
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=walker)
        return camera_bp, camera

    def _save_data(self, sensor_data):
        # 保存 RGB 图像到磁盘。
        file_path = os.path.join(f"{self.data_dir}/image/{self.camera_config['name']}", '%06d.jpg' % sensor_data.frame)
        sensor_data.save_to_disk(file_path)
        print(f"Saved RGB image to {file_path}")