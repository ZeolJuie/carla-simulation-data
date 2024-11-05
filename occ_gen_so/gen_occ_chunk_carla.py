import os
import yaml
import torch
import chamfer
import glob
import time
import open3d as o3d
import numpy as np
import pdb
 
from mmcv.ops.points_in_boxes import points_in_boxes_cpu 
from scipy.spatial.transform import Rotation
from copy import deepcopy
from utils import preprocess, create_mesh_from_map, point_transform_3d_batch

if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    # parse.add_argument('--data_path', type=str, default="/mnt/ws-data/data/project/SurroundOcc/tools/generate_occupancy_with_own_data/gt_generation_template")
    parse.add_argument('--data_path', type=str, default="/mnt/ws-data/data/project/occ_gen/carla_sync/template_record") # record_1730339723
    parse.add_argument('--out_path', type=str, default=None)
    parse.add_argument('--config_path', type=str, default='./carla_data/config.yaml')
    parse.add_argument('--len_sequence', type=int, default=20)  
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
    
    pc_dir = os.path.join(path, 'point_cloud')
    bbox_dir= os.path.join(path, '3d_bbox') 
    transform_infor_dir = os.path.join(path, 'transform_infor')
 
    pc_seman_pathes = sorted(glob.glob(os.path.join(pc_dir, "*.npy")))
    bbox_pathes = sorted(glob.glob(os.path.join(bbox_dir, "bbox_*.npy")))
    object_category_pathes = sorted(glob.glob(os.path.join(bbox_dir, "object_category_*.npy")))
    boxes_token_pathes = sorted(glob.glob(os.path.join(bbox_dir, "boxes_token_*.npy")))
    transform_infor_pathes = sorted(glob.glob(os.path.join(transform_infor_dir, "lidar_2_world_*.npy")))
     
    len_sequence = args.len_sequence
    len_side = args.len_side
     
    i = 0 
    all_sample_number = len(pc_seman_pathes)
    while i < all_sample_number:
         
        iter_start = time.time()
        
        dict_list = []
  
        left_side = i if i==0 else i - len_side
        right_side = left_side + len_sequence
     
        if right_side + len_sequence - len_side >= all_sample_number:
            right_side = all_sample_number
        print(f"process: {left_side}/{all_sample_number}")
        print(f"left_side: {left_side}, right_side: {right_side}")
        for j in range(left_side, right_side):
            # load data
            name = os.path.basename(pc_seman_pathes[j]).split('.')[0] 
            pc0 = np.load(pc_seman_pathes[j])
            if not args.with_semantic:
                pc0 = pc0[:, :3]
            boxes = np.load(bbox_pathes[j])
            object_category = np.load(object_category_pathes[j])
            boxes_token = np.load(boxes_token_pathes[j])
            lidar_2_world = np.load(transform_infor_pathes[j], allow_pickle=True)
            lidar_2_world_ref = np.load(transform_infor_pathes[i], allow_pickle=True)
            
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
            points_in_boxes = points_in_boxes_cpu(B, M)  

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
            object_points_dict[query_object_token] = []
            for dict in dict_list:
                for k, object_token in enumerate(dict['object_tokens']):
                    if query_object_token == object_token:  
                        object_points = dict['object_points_list'][k]  
                        if object_points.shape[0] > 0:
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
