import os
import sys
import csv
import time
import copy
import json
import math
import random
import argparse
from datetime import datetime
from queue import Queue
from queue import Empty

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import carla

import config
from sensors.camera import CameraSensor
from sensors.depth_camera import DepthCameraSensor
from sensors.semantic_camera import SemanticCameraSensor
from sensors.lidar import LidarSensor
from sensors.semantic_lidar import SemanticLidarSensor
from sensors.radar import RadarSensor
from sensors.imu import IMUSensor
from sensors.gnss import GNSSSensor
from utils.geometry_utils import *
from utils.folder_utils import *
from scenario.scenario_config import generate_scenario


def get_city_object_annotation(world, frame_id, object, walker):
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
        "Fences": carla.CityObjectLabel.Fences,
        "Other": carla.CityObjectLabel.Other,
        "Dynamic": carla.CityObjectLabel.Dynamic,
        "GuardRail": carla.CityObjectLabel.GuardRail,
    }

    # 通过Static 3D Bounding Box的体积，进一步判定类别

    bounding_box_set = world.get_level_bbs(label_mapping[object])

    for bb in bounding_box_set:

        # Filter for distance from ego vehicle
        if bb.location.distance(walker.get_transform().location) < config.VALID_DISTANCE:

            # Cycle through the vertices
            verts = [v for v in bb.get_world_vertices(carla.Transform())]

            # get center point of 3D box
            global_center = get_center_point(verts)

            # 获取 Bounding Box 的尺寸 (1/2size)
            bbox_extent = [bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2]
            
            rotation = bb.rotation
            global_rotation = [
                np.radians(rotation.roll),
                np.radians(rotation.pitch),
                np.radians(rotation.yaw)
            ]

            # 存储 Bounding Box 数据
            object_label = {
                'object_id': -1,                                    # 物体ID
                'class': object,                                    # 物体类别
                'truncation': 0.0,                                  # 截断程度（0-1）
                'occlusion': 0,                                     # 遮挡程度（0-3）
                'location': global_center,                          # in lidar coordinate(KITTI是相机坐标系下)
                'dimensions': bbox_extent,                          # 目标物体的尺寸
                'rotation': global_rotation                         # in lidar coordinate(KITTI是相机坐标系下)
            }

            object_labels.append(object_label)

    return object_labels


def main(main_folder: str, args):

    # parse args
    sequence_id = args.sequence
    scenario_id = args.scenario

    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    scenario = generate_scenario(scenario_id)
    
    world = client.get_world()

    # 使用CARLA预设天气（共9种预设）
    preset = scenario.weather
    world.set_weather(preset)
    
    try:
        # We need to save the settings to be able to recover them at the end of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 如果是自动控制，随机选择一个行人蓝图，并创建行人
        if not args.no_auto_controll:
            walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))

            # set start point and end point
            start_point = scenario.start_point
            end_point = scenario.end_point
            # end_point = world.get_random_location_from_navigation()

            try:
                # generate walker
                walker = world.spawn_actor(walker_bp, carla.Transform(start_point, carla.Rotation(yaw=180)))
            except:
                print("RuntimeError: Spawn failed because of collision at spawn position")
                world.apply_settings(original_settings)
                exit()
        else:
            assert args.walker_id is not None, "Walker ID must be provided!"
            walker = world.get_actor(actor_id=int(args.walker_id))
            walker.set_location(scenario.start_point)
            walker_blueprint_id = walker.type_id
            walker_bp = blueprint_library.find(walker_blueprint_id)

        # 生成若干行人npc
        pedestrians_num = 0 if args.no_auto_controll else 140
        spawn_points = []
        for i in range(pedestrians_num):
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point = carla.Transform(loc, carla.Rotation())
                spawn_points.append(spawn_point)
        
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
        
        print(f"{len(pedestrians)} was successfully spawned.")
        
        
        # create all the sensors and keep them in a list for convenience.
        sensor_list = []

        # import camera config
        camera_configs = config.camera_configs

        # create and set cameras
        for camera_config in camera_configs:
            camera = CameraSensor(world, blueprint_library, walker, main_folder, camera_config)
            sensor_list.append(camera)

        depth_camera = DepthCameraSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(depth_camera)
        semantic_camera = SemanticCameraSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(semantic_camera)
        lidar = LidarSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(lidar)
        semantic_lidar = SemanticLidarSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(semantic_lidar)
        radar = RadarSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(radar)
        imu = IMUSensor(world, blueprint_library, walker, main_folder)
        sensor_list.append(imu)
        gnss = GNSSSensor(world, blueprint_library, walker, main_folder)   
        sensor_list.append(gnss)

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queues = {sensor.sensor_type: Queue() for sensor in sensor_list}

        # 启动传感器
        for sensor in sensor_list:
            sensor.start(sensor_queues[sensor.sensor_type], sensor.sensor_type)
        
        # 创建行人 AI 控制器
        if not args.no_auto_controll:
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
        if not args.no_auto_controll:
            walker_controller.start()
            walker_controller.go_to_location(end_point)
            walker_controller.set_max_speed(1 + random.random())

        for controller in controllers:
            # 启动控制器，设置行人随机行走
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        all_actors = world.get_actors()
        for actor in all_actors:
            if "walker.pedestrian" in actor.type_id:
                actor.set_collisions(True)
                actor.set_simulate_physics(True)

        # 创建一个JSON结构（数组）存储标签数据，数组的每一个元素对应每一帧的标注数据
        labels = []

        # 创建pandas dataframe存储ego pose
        columns = [
            'frame', 'timestamp', 
            'location_x', 'location_y', 'location_z',
            'rotation_roll', 'rotation_pitch', 'rotation_yaw',
            'velocity_x', 'velocity_y', 'velocity_z'
        ]

        # 创建空的 DataFrame
        ego_pose_df = pd.DataFrame(columns=columns)

        map_name = world.get_map().name.split("/")[-1]  # 提取地图名称（去除路径）
        date_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds

        # 创建json文件存储log
        log_info = {
            'location': map_name,
            'date_captured': date_captured,
            'walker': {
                'age': walker_bp.get_attribute('age').as_str(),
                'gender': walker_bp.get_attribute('gender').as_str(),
                'role_name': walker_bp.get_attribute('role_name').as_str(),
                'ros_name': walker_bp.get_attribute('ros_name').as_str()
            },
            'scene': sequence_id,
            'description': '',
            'weather': '',
            'duration': 0,
            'frame_num': 0,
        }

        spec = world.get_spectator()

        # Main loop
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            w_timestamp = world.get_snapshot().timestamp.elapsed_seconds

            # Only save key frames
            if w_frame % config.INTER_FRAME != 0:
                continue

            log_info['frame_num'] += 1

            print(f"\n ----------- World's frame: {w_frame}: {w_timestamp} -----------")

            for sensor in sensor_list:
                try:
                    s_frame = sensor_queues[sensor.sensor_type].get(True, 0.5)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                except Exception:
                    print("raise Exception:", Exception)

            # 获取行人位置
            walker_location = walker.get_location()
            print(f"walker location: ({walker_location.x:.2f}, {walker_location.y:.2f}, {walker_location.z:.2f})")

            transform = walker.get_transform()
            velocity = walker.get_velocity()
            # 添加一行数据
            new_row = {
                'frame': '%06d' % w_frame,
                'timestamp': '%.12f' % w_timestamp,
                'location_x': transform.location.x,
                'location_y': transform.location.y,
                'location_z': transform.location.z,
                'rotation_roll': transform.rotation.roll,
                'rotation_pitch': transform.rotation.pitch,
                'rotation_yaw': transform.rotation.yaw,
                'velocity_x': velocity.x,
                'velocity_y': velocity.y,
                'velocity_z': velocity.z
            }

            # 使用 loc 添加行
            ego_pose_df.loc[len(ego_pose_df)] = new_row

            lidar_transform = semantic_lidar.get_transform()
            lidar_2_world = np.array(lidar_transform.get_matrix())

            transform_path = os.path.join(f"{main_folder}/velodyne_calib", '%06d.npy' % w_frame)
            np.save(transform_path, lidar_2_world)

            # 设置观察者的位置（行人后方 3 米，高度 2 米）
            spectator_location = walker_location + walker.get_transform().get_forward_vector() * -3.0 + carla.Location(z=2.0)  # 稍微抬高视角

            # 设置观察者的旋转（与行人同方向）
            spectator_rotation = walker.get_transform().rotation
            spectator_rotation.pitch = -20  # 稍微俯视（可选调整）

            # 更新 Spectator
            spec.set_transform(carla.Transform(spectator_location, spectator_rotation))
            
            try:

                # create a dict to save label for this frame (tick)
                # 
                frame_label = {
                    "frame_id": '%06d' % w_frame,
                    "objects": []
                }
           
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
                    if dist < config.VALID_DISTANCE:

                        # get the 3D coordinates of the bounding box in world coordinates
                        verts = [v for v in bbox.get_world_vertices(npc.get_transform())]
                        # get center point of 3D box
                        global_center = get_center_point(verts)

                        # 获取 Bounding Box 的尺寸 (1/2size)
                        bbox_extent = [bbox.extent.x * 2, bbox.extent.y * 2, bbox.extent.z * 2]

                        # 将世界坐标系旋转矩阵转角
                        npc_global_rotation = [
                            np.radians(transform.rotation.roll),
                            np.radians(transform.rotation.pitch),
                            np.radians(transform.rotation.yaw),
                        ]

                        npc_class = "Pedestrian" if "walker" in npc.type_id else (npc.attributes['base_type']).capitalize()
                        npc_class = "Car" if npc_class == '' else npc_class

                        # 存储 Bounding Box 数据
                        object_label = {
                            'object_id': npc.id,                                # 物体ID
                            'class': npc_class,                                 # 物体类别
                            'truncation': 0.0,                                  # 截断程度（0-1）
                            'occlusion': 0,                                     # 遮挡程度（0-3）
                            'location': global_center,                          # in global coordinate(KITTI是相机坐标系下/nuScenes在全局坐标系下)
                            'dimensions': bbox_extent,                          # 目标物体的尺寸
                            'rotation': npc_global_rotation                     # in global coordinate(KITTI是相机坐标系下/nuScenes在全局坐标系下)
                        }

                        # 在该帧的标签字典中添加该检测到的对象
                        frame_label["objects"].append(object_label)
                
                labels.append(frame_label)

                city_objects = [
                    "TrafficLight", 
                    "TrafficSigns", 
                    "Poles", 
                    "Static", 
                    "Fences",
                    "Other",
                    "Dynamic",
                    "GuardRail"
                ]
                for object in city_objects:
                    city_object_labels = get_city_object_annotation(
                        world = world, 
                        frame_id = w_frame, 
                        object = object,
                        walker = walker, 
                    )
                    frame_label["objects"].extend(city_object_labels)

            except Empty:
                print("    Some of the sensor information is missed")
        
    finally:

        # 保存 log info 为JSON文件
        end_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        log_info['duration'] = end_sim_time - start_sim_time
        log_file_path = f"{main_folder}/log.json"
        with open(log_file_path, 'w') as f:
            json.dump(log_info, f)
        print(f"Saved labels data to {log_file_path}")

        # 保存为 labels 为 JSON 文件
        file_path = f"{main_folder}/labels.json"
        with open(file_path, 'w') as f:
            json.dump(labels, f)
        print(f"Saved labels data to {file_path}")

        # 保存ego poses数组为csv
        ego_pose_df.to_csv(f"{main_folder}/ego.csv", index=False, sep=",")

        # 还原 client world 设置
        world.apply_settings(original_settings)

        # 销毁 actor
        if not args.no_auto_controll:
            walker.destroy()

        for controller in controllers:
            controller.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in pedestrians])
        client.apply_batch([carla.command.DestroyActor(x) for x in controllers])

        # destroy sensor actor in a single simulation step
        client.apply_batch([carla.command.DestroyActor(x.sensor) for x in sensor_list])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect Multimodel Data in CARLA World")
    parser.add_argument("sequence", type=str, help="Sequence_id")
    parser.add_argument("--scenario", type=str, help="Scenario_id")
    parser.add_argument("--no-auto-controll", action="store_true", default=False, help="Disable AI controller, default: False")
    parser.add_argument("--walker-id", type=str, help="If disable ai controller, must give a manual controll walker id")
    args = parser.parse_args()

    sequence_id = args.sequence
    main_folder = create_folders(config.DATA_DIR, sequence_id)

    try:
        main(main_folder, args)
    except KeyboardInterrupt:
        print(' - Exited by user.')