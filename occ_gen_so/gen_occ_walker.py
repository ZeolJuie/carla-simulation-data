import os
import csv
import yaml
import json
import torch
import chamfer
import glob
import time
import open3d as o3d
import numpy as np
import pdb

import carla
from mmcv.ops.points_in_boxes import points_in_boxes_cpu 
from scipy.spatial.transform import Rotation
from copy import deepcopy
from utils import preprocess, create_mesh_from_map, point_transform_3d_batch


def point_transform_3d(loc, M):
    """ 
        Transform a 3D point using a 4x4 matrix
    """
 
    point = np.array([loc.x, loc.y, loc.z, 1]) if isinstance(loc, carla.libcarla.Location) else np.array([loc[0], loc[1], loc[2], 1])
    point_transformed = np.dot(M, point)
    # normalize, 其实最后一位就是1.0
    point_transformed[0] /= point_transformed[3]
    point_transformed[1] /= point_transformed[3]
    point_transformed[2] /= point_transformed[3]
    return point_transformed[:3]

def world_to_lidar_obj(obj, lidar_to_world):
    """
    将世界坐标系下的物体转换到LiDAR坐标系
    
    参数:
        obj: dict - 世界坐标系下的物体信息（包含location和rotation）
        lidar_to_world: np.ndarray - 4x4的LiDAR到世界坐标系的变换矩阵
    
    返回:
        dict - LiDAR坐标系下的物体信息（保持原JSON格式）
    """
    # 提取世界坐标系下的位置和旋转
    world_loc = np.array([obj["location"][0], obj["location"][1], obj["location"][2]])
    world_rot = np.array(obj["rotation"])  # 欧拉角（通常为ZYX顺序）

    # 计算world_to_lidar的变换矩阵（求逆）
    world_to_lidar = np.linalg.inv(lidar_to_world)
    
    # 转换位置（齐次坐标）
    world_loc_hom = np.append(world_loc, 1.0)  # [x, y, z, 1]
    lidar_loc_hom = np.dot(world_to_lidar, world_loc_hom)
    lidar_loc = lidar_loc_hom[:3].tolist()  # 取前3维并转为list

    # 转换旋转（假设旋转是绕Z轴的偏航角yaw）
    # 方法：从变换矩阵中提取旋转部分，并调整欧拉角
    # 注意：这里简化处理，仅适用于3D旋转中的偏航角（Yaw）
    lidar_yaw = world_rot[2] - np.arctan2(lidar_to_world[1, 0], lidar_to_world[0, 0])
    # 保持roll和pitch为0
    lidar_rot = [0.0, 0.0, lidar_yaw]  

    # 构建输出对象（其他字段不变）
    lidar_obj = {
        "object_id": obj["object_id"],
        "class": obj["class"],
        "truncation": obj["truncation"],
        "occlusion": obj["occlusion"],
        "location": lidar_loc,
        "dimensions": obj["dimensions"], 
        "rotation": lidar_rot
    }
    
    return lidar_obj

def convert_bbox_format(original_bbox):
    """
    将原始边界框格式转换为标准格式 [x, y, z, dx, dy, dz, rz]
    
    参数:
        original_bbox (dict): 原始边界框数据，包含location(中心点), dimensions和rotation
    
    返回:
        np.ndarray: 转换后的边界框 [x, y, z, dx, dy, dz, rz]
    """
    # 提取原始数据
    location = original_bbox["location"]  # [x, y, z] (原始中心点坐标)
    dimensions = [
        original_bbox["dimensions"][0],
        original_bbox["dimensions"][1],
        original_bbox["dimensions"][2]
    ]# [dx, dy, dz] (长宽高)
    rotation_z = original_bbox["rotation"][2]  # 只需要z轴旋转角度
    
    # 计算底部中心坐标 (y轴向下调整半个高度)
    bottom_center = [
        location[0],  # x不变
        location[1],  # y不变
        # location[2] - dimensions[2]/2  # z向下移动半个高度
        location[2]
    ]
    
    # 组装为标准格式
    converted = bottom_center + dimensions + [rotation_z]
    
    return converted

def get_category_index(category):
    category_index_map = {
        'Bus': 15, 
        'Car': 14, 
        'Truck': 16, 
        'Motorcycle': 18,
        'Bicycle': 19, 
        'Pedestrian': 12,
        'Van': 15,
        }
    return category_index_map[category]

def get_extrinsic_matrix(x, y, z, roll, pitch, yaw):
    """
    根据平移和欧拉角生成4x4外参矩阵
    
    参数:
        x, y, z: 平移分量 (单位: 米)
        roll, pitch, yaw: 绕X/Y/Z轴的旋转角度 (单位: 弧度)
        (roll: X轴, pitch: Y轴, yaw: Z轴)
    
    返回:
        4x4 numpy数组表示的齐次变换矩阵
        格式:
        [[R11, R12, R13, x],
         [R21, R22, R23, y],
         [R31, R32, R33, z],
         [0,   0,   0,   1]]
    """

    # 计算旋转矩阵的各元素 (使用简写符号)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # 构建旋转矩阵 (Z-Y-X顺序，即yaw->pitch->roll)
    rotation_matrix = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr]
    ])
    
    # 构建4x4齐次变换矩阵
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = [x, y, z]
    
    return extrinsic

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为旋转矩阵。
    :param roll: 绕 X 轴的旋转角度（弧度）。
    :param pitch: 绕 Y 轴的旋转角度（弧度）。
    :param yaw: 绕 Z 轴的旋转角度（弧度）。
    :return: 3x3 旋转矩阵。
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx  # 旋转顺序：Z-Y-X
    return R

def calculate_cube_vertices(transform, rotation, dimension):
    """
    计算立方体的8个顶点坐标
    
    参数:
        transform (list/tuple/np.array): 立方体中心的平移向量 [x, y, z]
        rotation (list/tuple/np.array): 立方体的旋转角度 [rx, ry, rz] (弧度)
        dimension (list/tuple/np.array): 立方体的尺寸 [width, height, depth]
    
    返回:
        np.array: 8个顶点坐标的数组，形状为(8, 3)
    """
    # 转换为numpy数组
    transform = np.array(transform)
    rotation = np.array(rotation)
    dimension = np.array(dimension)
    
    # 立方体的局部坐标 (未旋转和平移前)
    half_dim = dimension / 2.0
    vertices_local = np.array([
        [-1, -1, -1],  # 0: 左前下
        [-1,  1, -1],  # 1: 右前下
        [ 1, -1, -1],  # 2: 右后下
        [ 1,  1, -1],  # 3: 左后下
        [-1, -1,  1],  # 4: 左前上
        [-1,  1,  1],  # 5: 右前上
        [ 1, -1,  1],  # 6: 右后上
        [ 1,  1,  1]   # 7: 左后上
    ]) * half_dim
    
    # 创建旋转矩阵 (绕x, y, z轴旋转)
    rx, ry, rz = rotation
    rotation_matrix = euler_to_rotation_matrix(rx, ry, rz)
    
    # 应用旋转和平移
    vertices_global = np.dot(vertices_local, rotation_matrix.T) + transform
    
    return vertices_global


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    # parse.add_argument('--data_path', type=str, default="/mnt/ws-data/data/project/SurroundOcc/tools/generate_occupancy_with_own_data/gt_generation_template")
    parse.add_argument('--data_path', type=str, default="/home/zhoumohan/codes/carla-simulation-data/carla_data/sequences/01") # record_1730339723
    parse.add_argument('--out_path', type=str, default=None)
    parse.add_argument('--config_path', type=str, default='./carla_data/config.yaml')
    parse.add_argument('--len_sequence', type=int, default=40)  
    parse.add_argument('--len_side', type=int, default=4) 
    parse.add_argument('--to_mesh', action='store_true', default=False)
    parse.add_argument('--with_semantic', action='store_true', default=True)
    parse.add_argument('--whole_scene_to_mesh', action='store_true', default=False)


    args=parse.parse_args()

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']
    
     
    path = args.data_path
    out_path = args.out_path or os.path.join(path, "occ")
    os.makedirs(out_path, exist_ok=True)
    
    # semantic point cloud path, N*3
    pc_dir = os.path.join(path, 'velodyne_semantic')
    transform_infor_dir =  os.path.join(path, 'velodyne_calib')

    # lidar to world = ego_to_world @ lidar_to_ego
    ego_transform_path = os.path.join(path, 'ego.csv')
 
    pc_seman_pathes = sorted(glob.glob(os.path.join(pc_dir, "*.bin")))
    
    # bounding boxes in world coordinate
    label_path = os.path.join(path, 'labels.json')

    transform_infor_pathes = sorted(glob.glob(os.path.join(transform_infor_dir, "*.npy")))

    # 'CATEGOTIES_INDEX': {'bus': 15, 'car': 14, 'truck': 16, 'motorcycle': 18,'bicycle': 19, 'Pedestrian': 13}
    # object_category_pathes = sorted(glob.glob(os.path.join(bbox_dir, "object_category_*.npy")))

    # object_id
    # boxes_token_pathes = sorted(glob.glob(os.path.join(bbox_dir, "boxes_token_*.npy")))
    # transform_infor_pathes = sorted(glob.glob(os.path.join(transform_infor_dir, "lidar_2_world_*.npy")))
     
    len_sequence = args.len_sequence
    len_side = args.len_side
     
    i = 0 
    all_sample_number = len(pc_seman_pathes)

    # 加载数据
    point_dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('cos_angle', np.float32), ('obj_idx', np.uint32), ('obj_tag', np.uint32)
    ])
    
    # load boxes
    with open(label_path, "r") as f:
        labels = json.load(f)
    labels = {frame["frame_id"]: frame['objects'] for frame in labels}
    

    lidar_to_ego = get_extrinsic_matrix(
        x = 0, y=0, z=1.0,
        roll=0, pitch=0, yaw=0
    )
    lidar_2_world_map = dict()
    with open(ego_transform_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for ego in reader:
            ego_transform = carla.Transform(
                location=carla.Location(
                    float(ego['location_x']),
                    float(ego['location_y']),
                    float(ego['location_z'])
                ),
                rotation=carla.Rotation(
                    roll=float(ego['rotation_roll']),
                    pitch=float(ego['rotation_pitch']),
                    yaw=float(ego['rotation_yaw'])
                )
            )

            ego_to_world = ego_transform.get_matrix()
            lidar_2_world_map[ego['frame']] = ego_to_world @ lidar_to_ego


    while i < all_sample_number:
        name_i = os.path.basename(pc_seman_pathes[i]).split('.')[0] 
         
        iter_start = time.time()
        
        dict_list = []
  
        left_side = i if i==0 else i - len_side
        right_side = left_side + len_sequence
     
        if right_side + len_sequence - len_side >= all_sample_number:
            right_side = all_sample_number
        print(f"process: {left_side}/{all_sample_number}")
        print(f"left_side: {left_side}, right_side: {right_side}")
        for j in range(left_side, right_side):
            # load data / frame
            name = os.path.basename(pc_seman_pathes[j]).split('.')[0] 

            # load point cloud
            points = np.fromfile(pc_seman_pathes[j], dtype=point_dtype)

            # process to N*3
            pc0 = np.column_stack([points['x'], -points['y'], points['z'], points['obj_tag']])
            instance_idx = points['obj_idx']

            if not args.with_semantic:
                pc0 = pc0[:, :3]
            
            lidar_2_world = lidar_2_world_map[name]
            
            lidar_2_world_ref = lidar_2_world_map[name_i]
            
            # # 第i帧的lidar_to_world
            # lidar_2_world_ref = list(lidar_2_world_map.values())[i]
            
            objects = labels[name]

            # transform to lidar coodinate 
            # dynamic_objects = [world_to_lidar_obj(obj, lidar_2_world) for obj in objects if 0 < obj['object_id'] < 10000]

            dynamic_objects = [obj for obj in objects if 0 < obj['object_id'] < 10000]

            boxes = []
            # box: [x, y, z, x_size, y_size, z_size, rz] in Lidar coordinate, (x, y, z) is the bottom center)
            # 读到的是世界坐标系下的Box -> 构建carla.Box
            for obj in dynamic_objects:
                verts = calculate_cube_vertices(
                    transform=obj["location"],
                    rotation=obj["rotation"],
                    dimension=obj["dimensions"]
                )

                world_2_lidar = np.linalg.inv(lidar_2_world)
                bbox_vert_lidar = [point_transform_3d(vert, world_2_lidar) for vert in verts]
                bbox_vert_lidar = np.array(bbox_vert_lidar)

                bottom_center = (bbox_vert_lidar[0, :] + bbox_vert_lidar[1, :] + bbox_vert_lidar[2, :] + bbox_vert_lidar[3, :]) / 4

                sz = np.array([obj["dimensions"][0], obj["dimensions"][1], obj["dimensions"][2]])
                rz = bbox_vert_lidar[1, :] - bbox_vert_lidar[0, :]
                rz = - np.array([np.arctan(-rz[0] / rz[1])])
                bbox_infor = np.concatenate((bottom_center, sz, rz))
                boxes.append(bbox_infor)

            boxes = np.array(boxes)
            
            # 'CATEGOTIES_INDEX': {'bus': 15, 'car': 14, 'truck': 16, 'motorcycle': 18,'bicycle': 19, 'Pedestrian': 13}
            object_category = [get_category_index(obj['class']) for obj in dynamic_objects]
            # OBJECT_ID
            boxes_token = [obj['object_id'] for obj in dynamic_objects]

            # def points_in_boxes_cpu(points: Tensor, boxes: Tensor) -> Tensor
            # Args:
            # points (torch.Tensor): [B, M, 3], [x, y, z] in
            #     LiDAR/DEPTH coordinate
            # boxes (torch.Tensor): [B, T, 7],
            #     num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            #     (x, y, z) is the bottom center.

            # Returns:
            # torch.Tensor: Return the box indices of points with the shape of
            # (B, M, T). Default background = 0.

            B = torch.from_numpy(pc0[:, :3][np.newaxis, :, :]) 
            M = torch.from_numpy(boxes[np.newaxis, :])
            M[0, :, 6] *= -1  
            # 1 * Points_Num * Object_Num
            # points_in_boxes = points_in_boxes_cpu(B, M)
            points_in_boxes = torch.zeros((1, pc0.shape[0], boxes.shape[0]), dtype=torch.int32)

            # 使用semantic lidar的obj_idx关联boxes_token, 进而得到points_in_boxes
            for k, box_token in enumerate(boxes_token):
                points_in_box = torch.from_numpy(instance_idx == box_token)
                points_in_boxes[0][:, k] = points_in_box

            object_points_list = []
 
            for k in range(0, points_in_boxes.shape[-1]): 
                object_points_mask = points_in_boxes[0][:, k].bool()
                object_points = pc0[object_points_mask]
                object_points_list.append(object_points)
         
            moving_mask = torch.ones_like(points_in_boxes)
            points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool() # points_in_boxes * moving_mask 和 points_in_boxes有啥区别
            points_mask = ~(points_in_boxes[0])  

            ############################# get point mask out of the vehicle itself ##########################
            ego_range = config['self_range']
            oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > ego_range[0]) |
                                            (np.abs(pc0[:, 1]) > ego_range[1]) |
                                            (np.abs(pc0[:, 2]) > ego_range[2]))
            ############################# get static scene segment ##########################
            points_mask = points_mask & oneself_mask 
            pc = pc0[points_mask] 
            
            ################## coordinate conversion to the same (first) LiDAR coordinate  ################## 
            world_2_lidar = np.linalg.inv(lidar_2_world)
            world_2_lidar_ref = np.linalg.inv(lidar_2_world_ref)
            lidar_2_lidar_ref = world_2_lidar_ref @ lidar_2_world
            lidar_ref_2_lidar = world_2_lidar @ lidar_2_world_ref
            
            # static point cloud to the first LiDAR coordinate
            lidar_pc = point_transform_3d_batch(pc[:, :3], lidar_2_lidar_ref)
            lidar_pc = np.concatenate((lidar_pc, pc[:, 3:]), axis=1)
            dict = {"object_tokens": boxes_token,
                    "object_points_list": object_points_list,
                    "lidar_pc": lidar_pc, 
                    "gt_bbox_3d": boxes,
                    "converted_object_category": object_category,
                    "lidar_2_lidar_ref": lidar_2_lidar_ref, 
                    "lidar_ref_2_lidar": lidar_ref_2_lidar,
                    "name": name}
            dict_list.append(dict)


        ################## concatenate all static scene segments  ########################
        lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]  
        lidar_pc = np.concatenate(lidar_pc_list, axis=0) 
    
        ################## concatenate all object segments (including non-key frames)  ########################
        object_token_zoo = []
        object_semantic = []
        for dict in dict_list:
            for k, object_token in enumerate(dict['object_tokens']):
                if object_token not in object_token_zoo: 
                    if (dict['object_points_list'][k].shape[0] > 0):  
                        object_token_zoo.append(object_token)
                        object_semantic.append(dict['converted_object_category'][k])
                    else:
                        continue

        object_points_dict = {}
        for query_object_token in object_token_zoo:
            # 将各帧的同一动态障碍物合并到一起
            object_points_dict[query_object_token] = []
            for dict in dict_list:
                for k, object_token in enumerate(dict['object_tokens']):
                    if query_object_token == object_token:  
                        object_points = dict['object_points_list'][k]  
                        if object_points.shape[0] > 0:
                            # objects points相对于 box bottom center的便宜
                            object_points[:, :3] = object_points[:, :3] - dict['gt_bbox_3d'][k][:3] 
                            rots = dict['gt_bbox_3d'][k][6]
                            Rot = Rotation.from_euler('z', -rots, degrees=False)  
                            object_points[:, :3] = Rot.apply(object_points[:, :3]) 
                            object_points_dict[query_object_token].append(object_points) 
                            
                    else:
                        continue
            object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                    axis=0)  

        object_points_vertice = [] 
        for key in object_points_dict.keys():
            point_cloud = object_points_dict[key]
            object_points_vertice.append(point_cloud)  
        
        if args.whole_scene_to_mesh:
            point_cloud_original = o3d.geometry.PointCloud()
            with_normal2 = o3d.geometry.PointCloud()
            point_cloud_original.points = o3d.utility.Vector3dVector(lidar_pc[:, :3])
            with_normal = preprocess(point_cloud_original, config)
            with_normal2.points = with_normal.points
            with_normal2.normals = with_normal.normals
            mesh, _ = create_mesh_from_map(None, 11, config['n_threads'],
                                        config['min_density'], with_normal2)
            lidar_pc = np.asarray(mesh.vertices, dtype=float)
            lidar_pc = np.concatenate((lidar_pc, np.ones_like(lidar_pc[:, 0:1])),axis=1)

        # TODO: 可视化发现，行人作为静态障碍物被合并到了同一坐标系下

        # points = lidar_pc[:, :3]  # xyz坐标
        # labels = lidar_pc[:, 3].astype(int)  # 语义标签
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # # pcd.colors = o3d.utility.Vector3dVector(colors)

        # # 可视化
        # o3d.visualization.draw_geometries(
        #     [pcd],
        #     window_name="LiDAR Point Cloud with Semantics",
        #     width=1024,
        #     height=768,
        # )
        # breakpoint()

        print("cloud merging done!")
        left_side_gen = 0
        right_side_gen = len(dict_list) if right_side==all_sample_number else len(dict_list) - len_side 
        print(f"left_side_gen: {left_side_gen}, right_side_gen: {right_side_gen}")
        for j in range(left_side_gen, right_side_gen):
            dict = dict_list[j]
            name = dict['name']
            ################## convert the static scene to the target coordinate system ##############
            lidar_ref_2_lidar = dict['lidar_ref_2_lidar']

            lidar_pc_i = point_transform_3d_batch(lidar_pc[:, :3], lidar_ref_2_lidar)
            lidar_pc_i = np.concatenate((lidar_pc_i, lidar_pc[:, 3:]), axis=1)
            point_cloud = lidar_pc_i[:, :3]
            if args.with_semantic:        
                point_cloud_with_semantic = lidar_pc_i[:,:4]
           
            gt_bbox_3d = dict['gt_bbox_3d']
            locs = gt_bbox_3d[:,0:3]
            dims = gt_bbox_3d[:,3:6]
            rots = gt_bbox_3d[:,6:7]
            # gt_bbox_3d[:, 2] += dims[:, 2] / 2.


            ################## bbox placement ##############
            object_points_list = []
            object_semantic_list = []
            for k, object_token in enumerate(dict['object_tokens']):
                for l, object_token_in_zoo in enumerate(object_token_zoo):
                    if object_token==object_token_in_zoo:
                        
                        points = deepcopy(object_points_vertice[l])  
                   
                        Rot = Rotation.from_euler('z', rots[k], degrees=False)  
                        rot_point = points[:, :3].copy()
                        rotated_object_points = Rot.apply(rot_point) 
                        points[:, :3] = rotated_object_points + locs[k] 
                    
                        if points.shape[0] >= 5:
                            points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]), torch.from_numpy(gt_bbox_3d[k:k+1][np.newaxis, :]))
                            points = points[points_in_boxes[0, :, 0].bool()]  
                 
                        object_points_list.append(points[:, :3])
                        # semantics = np.ones_like(points[:, 0:1]) * object_semantic[l]  
                        # object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))  
                      
                        object_semantic_list.append(points)  
  
            try: # avoid concatenate an empty array
                temp = np.concatenate(object_points_list)
                scene_points = np.concatenate([point_cloud, temp])  
                # scene_points = point_cloud  
            except:
                scene_points = point_cloud

            if args.with_semantic:
                try: 
                    temp = np.concatenate(object_semantic_list) 
                    # pdb.set_trace()
                    scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
                    # scene_semantic_points = point_cloud_with_semantic
                except:
                    scene_semantic_points = point_cloud_with_semantic
                
            ################## remain points with a spatial range ##############
            # mask = (np.abs(scene_points[:, 0]) < 50.0) & (np.abs(scene_points[:, 1]) < 50.0) \
            #        & (scene_points[:, 2] > -5.0) & (scene_points[:, 2] < 3.0) # 50, 5, 3 
            # scene_points = scene_points[mask]

            if args.to_mesh and not args.whole_scene_to_mesh:
                ################## get mesh via Possion Surface Reconstruction ##############
                point_cloud_original = o3d.geometry.PointCloud()
                with_normal2 = o3d.geometry.PointCloud()
                point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
                with_normal = preprocess(point_cloud_original, config)
                with_normal2.points = with_normal.points
                with_normal2.normals = with_normal.normals
                mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                            config['min_density'], with_normal2)
                scene_points = np.asarray(mesh.vertices, dtype=float)


            ################## remain points with a spatial range ##############
            mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
                & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
            scene_points = scene_points[mask]  

            ################## convert points to voxels ##############
            pcd_np = scene_points
            pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size  
            pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
            pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
            pcd_np = np.floor(pcd_np).astype(np.int32) 
            fov_voxels = np.unique(pcd_np, axis = 0).astype(np.float64)  
 
            ################## convert voxel coordinates to LiDAR system  ##############
            fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size # + 0.5 是 + voxel_size 吗, fov_voxels[:, :3] 不就是 fov_voxels吗
            fov_voxels[:, 0] += pc_range[0]
            fov_voxels[:, 1] += pc_range[1]
            fov_voxels[:, 2] += pc_range[2]        
            # 这里没看懂有啥区别
            # np.save(os.path.join(out_path, f'no_semantic_{name}.npy'), fov_voxels)

            if args.with_semantic:
                ################## remain points with a spatial range  ##############
                mask = (np.abs(scene_semantic_points[:, 0]) < pc_range[3]) & (np.abs(scene_semantic_points[:, 1]) < pc_range[4]) \
                    & (scene_semantic_points[:, 2] > pc_range[2]) & (scene_semantic_points[:, 2] < pc_range[5])
                scene_semantic_points = scene_semantic_points[mask]

                ################## Nearest Neighbor to assign semantics ##############
                dense_voxels = fov_voxels
                sparse_voxels_semantic = scene_semantic_points

                x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
                y = torch.from_numpy(sparse_voxels_semantic[:, :3]).cuda().unsqueeze(0).float()
    
                d1, d2, idx1, idx2 = chamfer.forward(x, y)
                indices = idx1[0].cpu().numpy()
 
                dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
                dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

                # to voxel coordinate
                pcd_np = dense_voxels_with_semantic
                pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
                pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
                pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
                dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int32) 
 
                dense_voxels_with_semantic[:, 1] *= -1 # convert to the nuscenes IMU coordinate
                semantics = np.zeros(occ_size) 
                
                pcd_np = np.floor(pcd_np).astype(np.int32)
                semantics[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = pcd_np[:, 3]  
                semantics = np.flip(semantics, axis=1).astype(np.int8) # coordinate: carla to nuscenes
                mask_camera = np.ones(occ_size).astype(np.int8) 
                mask_lidar = np.ones(occ_size).astype(np.int8)
                occ_out_dir = os.path.join(out_path, f'sample_{name}')
                dict_npz = {"semantics": semantics,  "mask_camera": mask_camera, "mask_lidar": mask_lidar}
                os.makedirs(occ_out_dir, exist_ok=True)
                np.savez(os.path.join(occ_out_dir, 'labels.npz'), **dict_npz)
                # np.save(os.path.join(occ_out_dir, 'labels.npy'), dense_voxels_with_semantic)
                 
        cost_time = time.time() - iter_start
        print(f'iter cost: {cost_time}s')
        i = right_side
