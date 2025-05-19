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

import config
from sensors.camera import CameraSensor
from sensors.depth_camera import DepthCameraSensor
from sensors.semantic_camera import SemanticCameraSensor
from sensors.lidar import LidarSensor
from sensors.radar import RadarSensor
from sensors.imu import IMUSensor
from sensors.gnss import GNSSSensor
from utils.geometry_utils import *
from utils.folder_utils import *


main_folder = create_folders(config.DATA_DIR, '04')

def get_city_object_annotation(world, frame_id, object, walker, w2l, w2c, K):
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
    label_mapping = {
        "TrafficLight": carla.CityObjectLabel.TrafficLight,
        "TrafficSigns": carla.CityObjectLabel.TrafficSigns,
        "Poles": carla.CityObjectLabel.Poles,
        "Static": carla.CityObjectLabel.Static,
    }

    bounding_box_set = world.get_level_bbs(label_mapping[object])

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
                
                # verts in world coordinate
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
                'class': object,                                    # 物体类别
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

        spawn_points = [world.get_random_location_from_navigation() for _ in range(300)]

        # sort x[-125, 120]   y[-80, 150]
        spawn_points = sorted(spawn_points, key=lambda point: (-point.x, -point.y))

        # set start point and end point

        # 02
        # start_point = carla.Location(x=118, y=-8.054474, z=0.158620)
        # end_point = carla.Location(x=80.076370, y=37.853725, z=0.158620)

        # 03
        start_point = carla.Location(x=27.677645, y=58.019924, z=0.158620)
        end_point = carla.Location(x=91.465294, y=81.790596, z=0.158620)

        # 生成行人
        walker = world.spawn_actor(walker_bp, carla.Transform(start_point, carla.Rotation()))

        # 生成npc
        pedestrians_num = 150
        # 获取所有可能的生成点
        spawn_points = []
        for i in range(pedestrians_num):
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point = carla.Transform(loc, carla.Rotation())
                spawn_points.append(spawn_point)
        
        # 生成300个行人
        pedestrians = []
        controllers = []
    
        for i in range(pedestrians_num):
            try:
                # 随机选择行人蓝图
                blueprint = random.choice(blueprint_library.filter('walker.pedestrian.*'))
                
                # 生成行人
                pedestrian = world.spawn_actor(blueprint, spawn_points[i])
                pedestrians.append(pedestrian)
                
            except Exception as e:
                print(f"Error spawning pedestrian {i}: {e}")
        

        # create all the sensors and keep them in a list for convenience.
        sensor_list = []

        # 创建并设置传感器
        camera = CameraSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(camera)
        depth_camera = DepthCameraSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(depth_camera)
        semantic_camera = SemanticCameraSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(semantic_camera)
        lidar = LidarSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(lidar)
        radar = RadarSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(radar)
        imu = IMUSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(imu)
        gnss = GNSSSensor(world, blueprint_library, walker, main_folder)   
        sensor_list.append(gnss)

        # 启动传感器
        camera.start(sensor_queue, "camera")
        depth_camera.start(sensor_queue, "depth")
        semantic_camera.start(sensor_queue, "semantic")
        lidar.start(sensor_queue, "lidar")
        radar.start(sensor_queue, "radar")
        imu.start(sensor_queue, "imu")
        gnss.start(sensor_queue, "gnss")
        
        # 创建行人 AI 控制器
        controller_bp = blueprint_library.find('controller.ai.walker')
        walker_controller = world.spawn_actor(controller_bp, carla.Transform(), walker)

        for pedestrian in pedestrians:
            # 为行人创建AI控制器
            controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
            controllers.append(controller)
       
        # important! wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 应用控制
        walker_controller.start()
        walker_controller.go_to_location(end_point)
        walker_controller.set_max_speed(1 + random.random())

        for controller in controllers:
            # 启动控制器，设置行人随机行走
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        spec = world.get_spectator()

        # Remember the edge pairs
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = camera.get_attribute("image_size_x").as_int()
        image_h = camera.get_attribute("image_size_y").as_int()
        fov = camera.get_attribute("fov").as_float()

        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # 创建一个JSON结构（数组）存储标签数据，数组的每一个元素对应每一帧的标注数据
        labels = []

        # Main loop
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame

            print("\n ----------- World's frame: %d -----------" % w_frame)

            # 获取行人位置
            walker_location = walker.get_location()
            print(f"walker location: ({walker_location.x:.2f}, {walker_location.y:.2f}, {walker_location.z:.2f})")


            # 设置观察者的位置（行人后方 3 米，高度 2 米）
            spectator_location = walker_location + walker.get_transform().get_forward_vector() * -3.0 + carla.Location(z=2.0)  # 稍微抬高视角

            # 设置观察者的旋转（与行人同方向）
            spectator_rotation = walker.get_transform().rotation
            spectator_rotation.pitch = -20  # 稍微俯视（可选调整）

            # 更新 Spectator
            spec.set_transform(carla.Transform(spectator_location, spectator_rotation))
            
            # spec.set_transform(carla.Transform(walker.get_transform().location + carla.Location(x=0.1, z=0.8), carla.Rotation()))

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
                        
                        npc_class = "Pedestrian" if "walker" in npc.type_id else npc.attributes['base_type']

                        # 存储 Bounding Box 数据
                        object_label = {
                            'object_id': npc.id,                                # 物体ID
                            'class': npc_class,                                 # 物体类别
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

                city_objects = ["TrafficLight", "TrafficSigns", "Poles", "Static"]
                for object in city_objects:
                    city_object_labels = get_city_object_annotation(
                        world = world, 
                        frame_id = w_frame, 
                        object = object,
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

        for controller in controllers:
            controller.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in pedestrians])
        client.apply_batch([carla.command.DestroyActor(x) for x in controllers])

        for sensor in sensor_list:
            sensor.destroy()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')