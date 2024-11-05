import queue
import carla    
import functools

import numpy as np 

from globals import get_global


class LidarManager:
    def __init__(self, sensor_config):
        self.sensor_config = sensor_config
        self.sensor_queues = [queue.Queue() for i in range(len(sensor_config))]
        self.sensor_list = []
        self.sensor_bp_list = []
        
    def spawn_sensor(self, world, vehicle=None):
        # spawn sensor
        delta = world.get_settings().fixed_delta_seconds
         
        for index, config in enumerate(self.sensor_config):
            sensor_bp = world.get_blueprint_library().find(config["sensor_type"])
            sensor_bp.set_attribute('rotation_frequency', str(1.0 / delta))
            for key, value in config["bp_attribute"].items():
                sensor_bp.set_attribute(key, str(value))
  
            sensor_init_trans = carla.Transform(carla.Location(**config["transform"]["location"]), carla.Rotation(**config["transform"]["rotation"]))
            sensor = world.spawn_actor(sensor_bp, sensor_init_trans, attach_to=vehicle)
            
            listen_func = functools.partial(self.semantic_lidar_callback, sensor_queue=self.sensor_queues[index])
            sensor.listen(listen_func)
            
            self.sensor_list.append(sensor)
            self.sensor_bp_list.append(sensor_bp)
            print('created %s' % sensor.type_id)
        return self.sensor_list
    
    def semantic_lidar_callback(self, sensor_data, sensor_queue):
        """Prepares a point cloud with semantic segmentation
        colors ready to be consumed by Open3D"""
        data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

        # # An example of adding some noise to our data if needed:
        # points += np.random.uniform(-0.05, 0.05, size=points.shape)

        # Unreal Engine中y轴和open3d中的是反的
        # 并且get_transform给出的变换矩阵现在看起来是以open3d中的为准的
        pc_semantic = np.array([data['x'], data['y'], data['z'], data['ObjTag']]).T
        sensor_queue.put(pc_semantic)
    
class CameraManager:
    def __init__(self, camera_config):
        self.camera_number = len(camera_config)
        self.camera_config = camera_config
        self.image_queues = [queue.Queue() for i in range(self.camera_number)]
        self.camera_list = []
        self.camera_bp_list = []

        self.K_list = []
        self.K_b_list = []

        self.camera_to_ego_list = []
        self.camera_to_lidar_list = []

        self.get_intrinsic_matrix()

        # 默认使用standard(nuscenes) camera坐标系时，需要先往carla的转
        self.camera_standard_to_carla = get_global('CAMERA_STANDARD_TO_CARLA')
        self.ego_carla_to_standard = get_global('EGO_CARLA_TO_STANDARD')
        self.lidar_carla_to_standard = get_global('LIDAR_CARLA_TO_STANDARD')

    def get_extrinsic_matrix(self, ref_transform, to_standard_ego=False, to_standard_lidar=False):
        camera_to_ref_list = []
        world_to_ref = ref_transform.get_inverse_matrix()
        # Get the extrinsic matrix from the camera to the world
        for camera in self.camera_list:
            camera_to_world = np.array(camera.get_transform().get_matrix())
            camera_to_ref = world_to_ref @ camera_to_world
            if to_standard_ego:
                camera_to_ref = self.ego_carla_to_standard @ camera_to_ref @ self.camera_standard_to_carla
            elif to_standard_lidar:
                camera_to_ref = self.lidar_carla_to_standard @ camera_to_ref @ self.camera_standard_to_carla
            camera_to_ref_list.append(camera_to_ref)

        return camera_to_ref_list

    def get_intrinsic_matrix(self):
        # Calculate the camera projection matrix to project from 3D -> 2D
        for i in range(len(self.camera_config)):
            # Get the attributes from the camera
            image_w = self.camera_config[i]["image_size"]["x"]
            image_h = self.camera_config[i]["image_size"]["y"]
            fov = self.camera_config[i]["fov"]

            K = self.build_projection_matrix(image_w, image_h, fov)
            K_b = self.build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)
            self.K_list.append(K)
            self.K_b_list.append(K_b)

    def spawn_camera(self, world, vehicle=None):
        # spawn camera
        for index, config in enumerate(self.camera_config):
            camera_bp = world.get_blueprint_library().find(config["sensor_type"])
            camera_bp.set_attribute('fov', str(config["fov"]))
            camera_bp.set_attribute('image_size_x', str(config["image_size"]["x"]))
            camera_bp.set_attribute('image_size_y', str(config["image_size"]["y"]))
            camera_init_trans = carla.Transform(carla.Location(**config["transform"]["location"]), carla.Rotation(**config["transform"]["rotation"]))
            camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

            listen_func = functools.partial(self.camera_show_callback, sensor_queue=self.image_queues[index], sensor_type="camera_rgb")
            camera.listen(listen_func)
            self.camera_list.append(camera)
            self.camera_bp_list.append(camera_bp)
            print('created %s' % camera.type_id)

        print(f'created {index+1} cameras total!')
        return self.camera_list

    def build_projection_matrix(self, w, h, fov, is_behind_camera=False):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)

        if is_behind_camera:
            K[0, 0] = K[1, 1] = -focal
        else:
            K[0, 0] = K[1, 1] = focal

        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def camera_show_callback(self, sensor_data, sensor_queue, sensor_type="sensor.camera.rgb"):
        array = np.frombuffer(np.copy(sensor_data.raw_data), dtype=np.dtype("uint8"))
        # image is rgba format
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4)) # r, g, b, a
        if "depth" in sensor_type:
            depth = array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * (256 * 256)
            depth = depth / (256 * 256 * 256 - 1) * 1000 # the default far plane is set at 1000 metres
            array = np.stack((depth, array[:, :, -1]), 2)
        sensor_queue.put((sensor_data.frame, array))
 