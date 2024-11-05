import sys
sys.path.append(".")
sys.path.append("..")
import carla
import random
import time 
import cv2.text
import argparse
import cv2
import os
import pdb 
import json
import pickle

import numpy as np
import open3d as o3d

from datetime import datetime 


from utils import *
from globals import get_global
from actor_manager import CameraManager, LidarManager


def main(arg):

    record_dir = f"./record_{time.time():.0f}"
    data_dir = "./data"
    camera_dir = os.path.join(record_dir, "camera")
    pkl_dir = os.path.join(record_dir, "pkl")
    data_template_path = os.path.join(data_dir, "pkl_template.pkl")
    config_path = "./configs/config.json"
    pc_dir = os.path.join(record_dir, "point_cloud")
    occ_dir = os.path.join(record_dir, "occ")
    transform_infor_dir = os.path.join(record_dir, "transform_infor")
    bbox_dir = os.path.join(record_dir, '3d_bbox')

    with open(config_path, "r") as f_config:
        config = json.load(f_config)
    with open(data_template_path, "rb") as f_data:
        data_template = pickle.load(f_data)
        data_template['infos'] = data_template['infos'][:1] # 只保留一个

    camera_config = config["camera"]
    lidar_config = config["lidar"]
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(occ_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(transform_infor_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    [os.makedirs(os.path.join(camera_dir, camera['direction']), exist_ok=True) for camera in camera_config]

    show_render = True
    save_data = True
    valid_distance = 50 # m
    delta = 1/10 # 系统帧率
    inter_frame = 5
    vehicle_number = 15
    frames_number = 100
    
    LABEL_COLORS = get_global('LABEL_COLORS')
    CATEGOTIES_INDEX = get_global('CATEGOTIES_INDEX')
    CATEGOTIES = get_global('CATEGOTIES')
    
    # set the world
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(120.0) # the max waiting time /s
    world  = client.get_world()
    # world = client.load_world('Town02')
    bp_lib = world.get_blueprint_library()
    # camera manager
    camera_manager = CameraManager(camera_config)
    # lidar manager 
    lidar_manager = LidarManager(lidar_config)
    try:
    # if True:
        original_settings = world.get_settings()
        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = delta
        settings.rendering_mode = arg.rendering
        world.apply_settings(settings)

        # Get the map spawn points
        spawn_points = world.get_map().get_spawn_points()

        # set the traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        print("set traffic manager done")

        # spawn vehicle
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        
        vehicle.set_autopilot(True)
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, -80) # fast driving
        print('created %s' % vehicle.type_id)

        # spawn more vehicle
        for i in range(vehicle_number):
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            # vehicle_bp = bp_lib.find('vehicle.harley-davidson.low_rider')
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
           
            if npc:
                npc.set_autopilot(True)

        # spawn camera
        camera_list = camera_manager.spawn_camera(world, vehicle)
  
        # spawn semantic lidar
        lidar_list = lidar_manager.spawn_sensor(world, vehicle)
   
        # Create the Open3D visualizer
        point_raw = o3d.geometry.PointCloud()
        point_warped = o3d.geometry.PointCloud()
        if show_render:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name='Carla Lidar',
                width=960,
                height=540,
                left=480,
                top=270)
            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 1
            vis.get_render_option().show_coordinate_frame = True

            if arg.show_axis:
                add_open3d_axis(vis)

        # Create the bounding box geometry
        line_sets = []
        color = [0.5, 0.5, 1.0]  # bbox为浅蓝色
        # Remember the edge pairs
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        # b: 6.------.4  t: 7.------.5
        #     |  ↑↑  |       |  ↑↑  |
        #     |      |       |      |
        #    2.------.0     3.------.1   右: y, 上: z, 前:x 
        if show_render:
            for i in range(5 + vehicle_number//2):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.zeros((12, 3), dtype=np.float32))
                line_set.lines = o3d.utility.Vector2iVector(edges)
                line_set.colors = o3d.utility.Vector3dVector([color] * len(line_set.lines))

                vis.add_geometry(line_set)
                line_sets.append(line_set)

        frame = 0
        save_index = 0
        dt0 = datetime.now()
        while frame < frames_number:
            # Retrieve and reshape the image
            world.tick()

            # save_index_str = f"{save_index:0>6d}" # 默认从000000开始
            time_stamp = f"{time.time():.2f}" # 时间戳只精确到小数点后两位
            time_stamp = time_stamp[:-3] + time_stamp[-2:] # 去掉小数点

            save_index_str = time_stamp # 使用时间戳来存，不用的话就去注释掉
            save_flag = save_data & (frame >= inter_frame) & (frame % inter_frame == 0)

            # save_pathes
            lidar_path = os.path.join(pc_dir, f"{save_index_str}.npy")
            lidar_o3d_path = os.path.join(pc_dir, f"{save_index_str}.ply")
            occ_path = os.path.join(occ_dir, f"sample_{save_index_str}") # a direction, 名字为sample_token
            pkl_path = os.path.join(pkl_dir, f"{save_index_str}.pkl")
            image_path = [os.path.join(camera_dir, camera['direction'], f"{save_index_str}.jpg") for camera in camera_config]
            bbox_path = os.path.join(bbox_dir, f"bbox_{save_index_str}.npy")
            boxes_token_path = os.path.join(bbox_dir, f"boxes_token_{save_index_str}.npy")
            object_category_path = os.path.join(bbox_dir, f"object_category_{save_index_str}.npy")
            transform_path = os.path.join(transform_infor_dir, f"lidar_2_world_{save_index_str}.npy")

            # rgb camera data 
            frame_index_and_images = [image_queue.get(True, 1.0) for image_queue in camera_manager.image_queues]

            # lidar(point cloud) data
            point_cloud = lidar_manager.sensor_queues[0].get(True, 1.0)

            points = point_cloud[:, :3].copy()
            labels = point_cloud[:, 3] 
            labels = np.clip(labels, 0, len(LABEL_COLORS) - 1).astype(np.int32)
            int_color = LABEL_COLORS[labels]

            point_raw.points = o3d.utility.Vector3dVector(points)
            point_raw.colors = o3d.utility.Vector3dVector(int_color)

            if show_render:
                if frame == 0:
                    vis.add_geometry(point_raw)
                vis.update_geometry(point_raw)

            # Get the transformed matrix
            lidar_transform = lidar_manager.sensor_list[0].get_transform()
            ego_transform = vehicle.get_transform()
 
            lidar_2_world = np.array(lidar_transform.get_matrix())
            world_2_lidar = np.array(lidar_transform.get_inverse_matrix())
            ego_2_world = np.array(ego_transform.get_matrix())
 
            if frame == 0:
                # get camera extrinsic matrix, to ego/lidar, once only
                scene_token = f"sence_{time_stamp}"
                world_2_ego = np.array(ego_transform.get_inverse_matrix())
                lidar_2_ego = world_2_ego @ lidar_2_world

                quat_lidar_2_ego  = matrix_to_quaternion(lidar_2_ego [:3, :3], to_wxyz=True)
                trans_lidar_2_ego  = lidar_2_ego[:3, 3]
                camera_manager.camera_to_ego_list = camera_manager.get_extrinsic_matrix(ego_transform, to_standard_ego=True)
                camera_manager.camera_to_lidar_list = camera_manager.get_extrinsic_matrix(lidar_transform, to_standard_lidar=True)
                rt_to_ego_list = []
                rt_to_lidat_list = []
                for k in range(camera_manager.camera_number):  

                    matrix_to_ego = camera_manager.camera_to_ego_list[k]
                    matrix_to_lidar = camera_manager.camera_to_lidar_list[k]
                    quat_to_ego = matrix_to_quaternion(matrix_to_ego[:3, :3], to_wxyz=True)
                    quat_to_lidar = matrix_to_quaternion(matrix_to_lidar[:3, :3], to_wxyz=True)
                    trans_to_ego = matrix_to_ego[:3, 3]
                    trans_to_lidar = matrix_to_lidar[:3, 3]
                    rt_to_ego_list.append((quat_to_ego, trans_to_ego))
                    rt_to_lidat_list.append((quat_to_lidar, trans_to_lidar))

            # get all actors
            motorcycle = world.get_level_bbs(carla.CityObjectLabel.Motorcycle)
            bicycle = world.get_level_bbs(carla.CityObjectLabel.Bicycle)
            bus = world.get_level_bbs(carla.CityObjectLabel.Bus)
            car = world.get_level_bbs(carla.CityObjectLabel.Car)
            truck = world.get_level_bbs(carla.CityObjectLabel.Truck)
            # 这里carla有bug，get_level_bbs()取到的actor两轮车厚度是0
            motorcycle = [bbox for bbox in motorcycle if bbox.extent.x * bbox.extent.y * bbox.extent.z != 0]
            bicycle = [bbox for bbox in bicycle if bbox.extent.x * bbox.extent.y * bbox.extent.z != 0]
            # actor两轮车的bbox使用get_actors()取到的
            motor_actors_list =  world.get_actors().filter('*vehicle*')
            motor_2_wheels = [motor for motor in motor_actors_list if motor.attributes['number_of_wheels'] == '2']
            motorcycle = [motor for motor in motor_2_wheels if motor.attributes['base_type'] == 'motorcycle'] + motorcycle
            bicycle = [motor for motor in motor_2_wheels if motor.attributes['base_type'] == 'bicycle'] + bicycle
            vehicles = bus + car + truck + motorcycle + bicycle
            categories = [CATEGOTIES_INDEX['bus']] * len(bus) + [
                CATEGOTIES_INDEX['car']
            ] * len(car) + [CATEGOTIES_INDEX['truck']] * len(truck) + [
                CATEGOTIES_INDEX['motorcycle']
            ] * len(motorcycle) + [CATEGOTIES_INDEX['bicycle']] * len(bicycle)

            valid_vehicle_number = 0
            bboxes_infor = []
            tokens_infor = []
            categories_infor = [] 
            for token, npc in enumerate(vehicles):
              
                bbox = npc if isinstance(npc, carla.BoundingBox) else npc.bounding_box
                transfom = carla.Transform() if isinstance(npc, carla.BoundingBox) else npc.get_transform()
                bbox_location = bbox.location if isinstance(npc, carla.BoundingBox) else npc.get_transform().location

                # Filter out the ego vehicle (seems no use here)
                if bbox_location != vehicle.get_transform().location:
                    # Fixing bug related to Carla 9.11 onwards where some bounding boxes have 0 extent
                    buggy_bbox = (bbox.extent.x * bbox.extent.y * bbox.extent.z == 0)
                    if buggy_bbox:
                        print("buggy bbox!")
                        continue

                    if bbox_location.distance(vehicle.get_transform().location) < valid_distance:
                   
                        # Get the 3D bounding box vertices
                        bbox_vert_world = [v for v in bbox.get_world_vertices(transfom)] # 转到世界坐标下

                        bbox_vert_lidar = [point_transform_3d(vert, world_2_lidar) for vert in bbox_vert_world] # 转到雷达坐标系下
                        bbox_vert_lidar = np.array(bbox_vert_lidar)

                        bottom_center = (bbox_vert_lidar[0, :] + bbox_vert_lidar[2, :] + bbox_vert_lidar[4, :] + bbox_vert_lidar[6, :]) / 4

                        sz = np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z]) * 2
                        rz = bbox_vert_lidar[2, :] - bbox_vert_lidar[0, :] # 原来3 - 0
                        rz = - np.array([np.arctan(-rz[0] / rz[1])]) # - arctan(x/y) , 原来无 -

                        bbox_infor = np.concatenate((bottom_center, sz, rz))
                        bboxes_infor.append(bbox_infor)
                        tokens_infor.append(token)
                        categories_infor.append(categories[token])

                        points = o3d.utility.Vector3dVector(bbox_vert_lidar)
            
                        if show_render and (frame > 0) and (valid_vehicle_number < len(line_sets)):

                            line_sets[valid_vehicle_number].points = o3d.utility.Vector3dVector(points)
                            vis.update_geometry(line_sets[valid_vehicle_number])
                  
                        valid_vehicle_number += 1
               
            # lidar to ego, only save once
            if frame == inter_frame:
                world_2_lidar_ref = world_2_lidar

            # data saving
            if save_flag:
                # save box information
                np.save(bbox_path, np.stack(bboxes_infor))
                np.save(boxes_token_path, np.array(tokens_infor))
                np.save(object_category_path, np.array(categories_infor))

                # save camera and lidar data
                for index, frame_index_and_image in enumerate(frame_index_and_images):
                    cv2.imwrite(image_path[index], frame_index_and_image[1])
                np.save(lidar_path, point_cloud)
                # 雷达点云可视化文件存储
                # o3d.io.write_point_cloud(lidar_o3d_path, point_raw)

                # save transform informations
                np.save(transform_path, lidar_2_world)

                data_pkl = data_template.copy()
                data_pkl['infos'][0]['lidar_path'] = lidar_path # TODO: 记得换nuscenes的坐标系
                data_pkl['infos'][0]['occ_path'] = occ_path # TODO: 记得换nuscenes的坐标系

                bboxes_infor = np.stack(bboxes_infor)
                # bbox lidar coodinate from carla to nuscenes
                bboxes_infor[:, [0, 1]] = bboxes_infor[:, [1, 0]]
                bboxes_infor[:, -1] *= -1
                data_pkl['infos'][0]['gt_boxes'] = bboxes_infor # x y z l w h rz, under the lidar coodinate
                categories_infor_str = np.array([np.str_(CATEGOTIES[index]) for index in categories_infor])
                data_pkl['infos'][0]['gt_names'] = categories_infor_str # np.str_('category_name')
                data_pkl['infos'][0]['timestamp'] = int(time_stamp)
                data_pkl['infos'][0]['token'] = f'sample_{time_stamp}'
                data_pkl['infos'][0]['scene_token'] = scene_token
                data_pkl['infos'][0]['scene_name'] = scene_token # 暂时用同一个
                ego_2_world_quat = matrix_to_quaternion(ego_2_world[:3, :3], to_wxyz=True)
                ego_2_world_trans = ego_2_world[:3, 3]
                data_pkl['infos'][0]['ego2global_rotation'] = ego_2_world_quat
                data_pkl['infos'][0]['ego2global_translation'] = ego_2_world_trans
                data_pkl['infos'][0]['lidar2ego_rotation'] = quat_lidar_2_ego
                data_pkl['infos'][0]['lidar2ego_translation'] = trans_lidar_2_ego
                for k, camera in enumerate(camera_config):
                    data_pkl['infos'][0]['cams'][camera['direction']]['data_path'] = image_path[k]
                    camera_type = camera['direction']
                    data_pkl['infos'][0]['cams'][camera['direction']]['type'] = camera_type
                    data_pkl['infos'][0]['cams'][camera['direction']]['sample_data_token'] = f'{camera_type}_{time_stamp}'
                    data_pkl['infos'][0]['cams'][camera['direction']]['sensor2ego_rotation'] = rt_to_ego_list[k][0]
                    data_pkl['infos'][0]['cams'][camera['direction']]['sensor2ego_translation'] = rt_to_ego_list[k][1]
                    data_pkl['infos'][0]['cams'][camera['direction']]['sensor2lidar_rotation'] = rt_to_lidat_list[k][0]
                    data_pkl['infos'][0]['cams'][camera['direction']]['sensor2lidar_translation'] = rt_to_lidat_list[k][1]
                    data_pkl['infos'][0]['cams'][camera['direction']]['ego2global_rotation'] = ego_2_world_quat
                    data_pkl['infos'][0]['cams'][camera['direction']]['ego2global_translation'] = ego_2_world_trans
                    data_pkl['infos'][0]['cams'][camera['direction']]['timestamp'] = int(time_stamp)
                    data_pkl['infos'][0]['cams'][camera['direction']]['cam_intrinsic'] = camera_manager.K_list[k]
                    # 完善cams的信息

                with open(pkl_path, "wb") as f_pkl:
                    pickle.dump(data_pkl, f_pkl)

                # warp point cloud and save, for debug
                # points= point_cloud[:, :3].copy()
                # labels = point_cloud[:, 3]
                # transform_matrixs = ego_2_lidar_ref @ world_2_ego_ref @ ego_2_world @ lidar_2_ego
                # transform_matrixs = world_2_lidar_ref @ lidar_2_world
                # points_warped = point_transform_3d_batch(points, transform_matrixs)
                # labels = np.clip(labels, 0, len(LABEL_COLORS) - 1).astype(np.int32)
                # int_color = LABEL_COLORS[labels]
                # point_warped.points = o3d.utility.Vector3dVector(points_warped)
                # point_warped.colors = o3d.utility.Vector3dVector(int_color)
                # o3d.io.write_point_cloud(os.path.join(pc_dir, f"pc_seman_{save_index}_warped.ply"), point_warped)

                # transform_matrixs = lidar_2_world
                # points_warped = point_transform_3d_batch(points, transform_matrixs)
                # point_warped.points = o3d.utility.Vector3dVector(points_warped)
                # point_warped.colors = o3d.utility.Vector3dVector(int_color)
                # o3d.io.write_point_cloud(os.path.join(pc_dir, f"pc_seman_{save_index}_warped_world.ply"), point_warped)

                if isinstance(save_index, int):
                    save_index += 1

            print("valid_vehicle_number: ", valid_vehicle_number)
            if show_render:
                vis.poll_events()
                vis.update_renderer()
                # cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('ImageWindowName', frame_index_and_images[0][1])
                # cv2.waitKey(1)

                for i in range(len(line_sets) - 1):
                    line_sets[i].points = o3d.utility.Vector3dVector(np.zeros((12, 3), dtype=np.float32))
                    vis.update_geometry(line_sets[i])

            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            # process_time = datetime.now() - dt0
            # sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            # sys.stdout.flush()
            # dt0 = datetime.now()

            frame += 1
    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
       
        vehicles = world.get_actors().filter('vehicle.*')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
        print('\ndestroying %d vehicles' % len(vehicles))
        sensors = world.get_actors().filter('sensor.*')
        client.apply_batch([carla.command.DestroyActor(x) for x in sensors])
        print('\ndestroying %d sensors' % len(sensors))
     
        if show_render:
            vis.destroy_window()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--rendering',
        action='store_true',
        default=True,
        help='use the rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--show_axis',
        default=True,
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    
    args = argparser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
