import os
import sys
import time
import copy
import json
import random
from datetime import datetime
from queue import Queue
from queue import Empty

import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import carla

from utils.geometry_utils import *
from utils.folder_utils import *


main_folder = create_folders('./carla_data', '04')


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    try:
        if sensor_name == "camera":
            # 处理 RGB 图像
            save_rgb_image(sensor_data)
        elif sensor_name == "lidar":
            # 处理 LiDAR 点云
            save_lidar_point_cloud(sensor_data)
        elif sensor_name == "imu":
            # 处理 IMU 数据
            save_imu_data(sensor_data)
        
        # 将数据帧和传感器名称放入队列
        sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))
    except Exception as e:
        print(f"Error processing {sensor_name} data: {e}")


def save_rgb_image(sensor_data):
    """
    保存 RGB 图像到磁盘。
    """
    file_path = os.path.join(f"{main_folder}/image", '%06d.png' % sensor_data.frame)
    sensor_data.save_to_disk(file_path)
    print(f"Saved RGB image to {file_path}")


def save_lidar_point_cloud(sensor_data):
    """
    保存 LiDAR 点云到磁盘。
    """
    data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype("f4")))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    points = data[:, :-1]
    # TODO: 这里为什么取负号？
    points[:, 1] = -points[:, 1]

    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)

    file_path = os.path.join(f"{main_folder}/velodyne", '%06d.ply' % sensor_data.frame)
    o3d.io.write_point_cloud(file_path, o3d_point_cloud)

    print(f"Saved LiDAR point cloud to {file_path}")


def save_imu_data(sensor_data):
    """
    保存 IMU 数据到磁盘。
    """
    print(f"Saved IMU data")


def get_city_object_annotation(world, frame_id, filter, walker, w2l, w2c, K):
    """
    获取某一帧下的actor的标注数据
    :world 
    :param actor_filter
    :param frame_id 帧ID
    :param walker 行人对象
    :param w2l 当帧下世界坐标系到雷达坐标系的转换矩阵, 用于雷达坐标系下的3D Box标注
    :param w2c 当帧下世界坐标系到相机坐标系的转换矩阵, 用于相机坐标系下的2D Box标注
    :return object_labels 对象标签列表
    """

    object_labels = []

    """
        添加交通灯/静态障碍物的标注
        For objects in the map like buildings, traffic lights and road signs, 
        the bounding box can be retrieved through the carla.World method get_level_bbs()
    """

    # Retrieve all bounding boxes for traffic lights within the level
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.Static))

    for bb in bounding_box_set:

        # Filter for distance from ego vehicle
        if bb.location.distance(walker.get_transform().location) < 50:

            # Cycle through the vertices
            verts = [v for v in bb.get_world_vertices(carla.Transform())]

            # get center point of 3D box
            global_center = get_center_point(verts)
            global_center = np.array([global_center[0], global_center[1], global_center[2], 1])

            # world coordinate -> lidar coordinate
            lidar_bbox_center = np.dot(w2l, global_center)

            # 获取 Bounding Box 的尺寸 (1/2size)
            bbox_extent = [bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2]

            # 将世界坐标系旋转矩阵转换到传感器坐标系
            rotation = bb.rotation
            
            npc_global_rotation = euler_to_rotation_matrix(
                np.radians(rotation.roll),
                np.radians(rotation.pitch),
                np.radians(rotation.yaw)
            )

            R_w2l = w2l[:3, :3]
            
            # npc 在雷达坐标系下的旋转矩阵
            lidar_rotation_matrix = np.dot(np.linalg.inv(R_w2l), npc_global_rotation)

            # npc 在雷达坐标系下的欧拉角
            lidar_rotation_euler = rotation_matrix_to_euler(lidar_rotation_matrix)


            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = walker.get_transform().get_forward_vector()
            ray = bb.location - walker.get_transform().location

            x_max = -10000
            x_min = 10000
            y_max = -10000
            y_min = 10000

            if forward_vec.dot(ray) > 0:
                p1 = get_image_point(bb.location, K, w2c)
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                

                for vert in verts:
                    p = get_image_point(vert, K, w2c)
                    # Find the rightmost vertex
                    if p[0] > x_max:
                        x_max = p[0]
                    # Find the leftmost vertex
                    if p[0] < x_min:
                        x_min = p[0]
                    # Find the highest vertex
                    if p[1] > y_max:
                        y_max = p[1]
                    # Find the lowest  vertex
                    if p[1] < y_min:
                        y_min = p[1]

            # 存储 Bounding Box 数据
            object_label = {
                'object_id': -1,                                    # 物体ID
                'class': 'Static',                                  # 物体类别
                'truncation': 0.0,                                  # 截断程度（0-1）
                'occlusion': 0,                                     # 遮挡程度（0-3）
                'bbox': [x_min, y_min, x_max, y_max],               # 2D Bounding Box [x1, y1, x2, y2]
                'location': lidar_bbox_center[0: 3].tolist(),       # in lidar coordinate(KITTI是相机坐标系下)
                'dimensions': bbox_extent,                          # 目标物体的尺寸
                'rotation': lidar_rotation_euler.tolist()           # in lidar coordinate(KITTI是相机坐标系下)
            }
        
            object_labels.append(object_label)

    return object_labels


def main():

    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # traffic_manager = client.get_trafficmanager()
        # traffic_manager.set_synchronous_mode(True)


        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()

        blueprint_library = world.get_blueprint_library()

        # 随机选择一个行人蓝图
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))

        # 获取地图的推荐点
        spawn_points = world.get_map().get_spawn_points()

        # 随机选择一个生成点
        # spawn_point = random.choice(world.get_map().get_spawn_points())
        spawn_point = world.get_random_location_from_navigation()

        # 生成行人
        walker = world.spawn_actor(walker_bp, random.choice(spawn_points))

        # create all the sensors and keep them in a list for convenience.
        sensor_list = []

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        imu_bp = blueprint_library.find('sensor.other.imu')

        # 设置摄像头
        camera_bp.set_attribute('image_size_x', '1600')
        camera_bp.set_attribute('image_size_y', '1200')
        camera_bp.set_attribute('fov', '120')
        # camera_bp.set_attribute('sensor_tick', '0.1')  # 设置摄像头频率为 10 Hz
        camera_transform = carla.Transform(carla.Location(x=0.0, z=1.5))  # 摄像头位置
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=walker)

        camera.listen(lambda data: sensor_callback(data, sensor_queue, "camera"))
        sensor_list.append(camera)


        # 设置激光雷达
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

        lidar_bp.set_attribute("upper_fov", str(15.0))
        lidar_bp.set_attribute("lower_fov", str(-25.0))
        lidar_bp.set_attribute("channels", str(64.0))
        lidar_bp.set_attribute("range", str(100.0))
        lidar_bp.set_attribute("rotation_frequency", str(20.0 / 1))
        lidar_bp.set_attribute("points_per_second", str(1000000))

        # 激光雷达位置
        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=walker)

        lidar.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))
        sensor_list.append(lidar)

        # imu同雷达绑定
        imu_transform = lidar_transform
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=walker)
        
        imu.listen(lambda data: sensor_callback(data, sensor_queue, "imu"))
        sensor_list.append(imu)

        # 创建行人 AI 控制器
        controller_bp = blueprint_library.find('controller.ai.walker')
        walker_controller = world.spawn_actor(controller_bp, carla.Transform(), walker)

        # important! wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 应用控制
        walker_controller.start()
        walker_controller.go_to_location(world.get_random_location_from_navigation())
        walker_controller.set_max_speed(1 + random.random())
        spec = world.get_spectator()

        # Remember the edge pairs
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # 创建一个JSON结构（数组）存储标签数据，数组的每一个元素对应每一帧的标注数据
        labels = []

        # Main loop
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # 获取行人位置
            walker_location = walker.get_location()
            print(f"walker location: ({walker_location.x:.2f}, {walker_location.y:.2f}, {walker_location.z:.2f})")
            
            spec.set_transform(carla.Transform(walker.get_transform().location + carla.Location(z=10), carla.Rotation(pitch=-90)))

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.

            try:

                # create a dict to save label for this frame (tick)
                frame_label = {
                    "frame_id": '%06d' % w_frame,
                    "objects": []
                }
                
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

                # Get the camera matrix 
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

                world_2_lidar =  np.array(lidar.get_transform().get_inverse_matrix())

                # !NOTICE: get_transform().rotation - 角度制
                # 传感器的旋转矩阵 : 世界坐标系 到 Lidar坐标系 的旋转矩阵
                R_w2l = world_2_lidar[:3, :3]

                # 设置一个标注的filter，方便某一次采集只标注车辆/行人
                filters = ['*pedestrian*', '*vehicle*']

                npcs = []
                for filter in filters:
                    npcs.extend([p for p in world.get_actors().filter(filter)])

                for npc in npcs:

                    # 获取3D Bounding Box
                    bbox = npc.bounding_box

                    transform = npc.get_transform()

                    # 计算 walker 和 npc 之间的距离
                    dist = transform.location.distance(walker.get_transform().location)

                    # 筛选距离在50米以内的车辆
                    if dist < 50:

                        # get the 3D coordinates of the bounding box in world coordinates
                        verts = [v for v in bbox.get_world_vertices(npc.get_transform())]
                        # get center point of 3D box
                        global_center = get_center_point(verts)
                        global_center = np.array([global_center[0], global_center[1], global_center[2], 1])

                        # world coordinate -> lidar coordinate
                        lidar_bbox_center = np.dot(world_2_lidar, global_center)

                        # 获取 Bounding Box 的尺寸 (1/2size)
                        bbox_extent = [bbox.extent.x * 2, bbox.extent.y * 2, bbox.extent.z * 2]

                        # 将世界坐标系旋转矩阵转换到传感器坐标系
                        rotation = transform.rotation
                        npc_global_rotation = np.array(transform.get_inverse_matrix())[:3, :3]
                        
                        # npc 在雷达坐标系下的旋转矩阵
                        lidar_rotation_matrix = np.dot(np.linalg.inv(R_w2l), npc_global_rotation)

                        # npc 在雷达坐标系下的欧拉角
                        lidar_rotation_euler = rotation_matrix_to_euler(lidar_rotation_matrix)

                        """
                            get 2D Bounding Box
                        """

                        forward_vec = walker.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - walker.get_transform().location

                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        if forward_vec.dot(ray) > 0:
                            p1 = get_image_point(bbox.location, K, world_2_camera)
                            
                            for vert in verts:
                                p = get_image_point(vert, K, world_2_camera)
                                # Find the rightmost vertex
                                if p[0] > x_max:
                                    x_max = p[0]
                                # Find the leftmost vertex
                                if p[0] < x_min:
                                    x_min = p[0]
                                # Find the highest vertex
                                if p[1] > y_max:
                                    y_max = p[1]
                                # Find the lowest  vertex
                                if p[1] < y_min:
                                    y_min = p[1]

                        # 存储 Bounding Box 数据
                        object_label = {
                            'object_id': npc.id,                                # 物体ID
                            'class': npc.type_id,                               # 物体类别
                            'truncation': 0.0,                                  # 截断程度（0-1）
                            'occlusion': 0,                                     # 遮挡程度（0-3）
                            'bbox': [x_min, y_min, x_max, y_max],               # 2D Bounding Box [x1, y1, x2, y2]
                            'location': lidar_bbox_center[0: 3].tolist(),       # in lidar coordinate(KITTI是相机坐标系下)
                            'dimensions': bbox_extent,                          # 目标物体的尺寸
                            'rotation': lidar_rotation_euler.tolist()           # in lidar coordinate(KITTI是相机坐标系下)
                        }

                        # 在该帧的标签字典中添加该检测到的对象
                        frame_label["objects"].append(object_label)
                
                labels.append(frame_label)

                city_object_labels = get_city_object_annotation(
                    wolrd = world, 
                    frame_id = w_frame, 
                    filter = [carla.CityObjectLabel.TrafficLight, carla.CityObjectLabel.Static], 
                    walker = walker, 
                    w2l = world_2_lidar, 
                    w2c = world_2_camera, 
                    K = K
                )
                frame_label["objects"].extend(city_object_labels)

            except Empty:
                print("    Some of the sensor information is missed")
        
    finally:

        # 保存为 labels 为 JSON 文件
        file_path = f"{main_folder}/labels.json"
        with open(file_path, 'w') as f:
            json.dump(labels, f)
        print(f"Saved labels data to {file_path}")

        # 还原 client world 设置
        world.apply_settings(original_settings)

        # 销毁 actor
        walker.destroy()

        for sensor in sensor_list:
            sensor.destroy()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')