# sensors/camera.py

import os
import time
import queue

import carla

import config

class CameraSensor:
    def __init__(self, world, blueprint_library, walker, data_dir):
        self.world = world
        self.data_dir = data_dir
        self.camera = self._setup_camera(blueprint_library, walker)

    def _setup_camera(self, blueprint_library, walker):
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.CAMERA_IMAGE_SIZE_X))
        camera_bp.set_attribute('image_size_y', str(config.CAMERA_IMAGE_SIZE_Y))
        camera_bp.set_attribute('fov', str(config.CAMERA_FOV))
        camera_bp.set_attribute('sensor_tick', str(config.CAMERA_SENSOR_TICK))
        camera_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=walker)
        return camera

    def camera_callback(self, image, image_queue):
        timestamp = int(time.time() * 1000)
        image.save_to_disk(os.path.join(self.data_dir, f"camera_{timestamp}.png"))
        print(f"保存图像: camera_{timestamp}.png")

    def save_to_disk(self, image):
        timestamp = int(time.time() * 1000)
        image.save_to_disk(os.path.join(self.data_dir, f"camera_{timestamp}.png"))
        print(f"保存图像: camera_{timestamp}.png")

    def sensor_callback(self, sensor_data, sensor_queue, sensor_name):
        # Do stuff with the sensor_data data like save it to disk
        # Then you just need to add to the queue
        sensor_queue.put((sensor_data.frame, sensor_name))

    def start(self, sensor_queue, sensor_name):
        self.camera.listen(lambda data: self.sensor_callback(data, sensor_queue, sensor_name))

    def destroy(self):
        if self.camera:
            self.camera.destroy()