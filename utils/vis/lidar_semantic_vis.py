import open3d as o3d
import numpy as np
import carla

import argparse
import json
import glob
import time
import os
import sys
import csv
sys.path.append('.')

from utils.geometry_utils import calculate_cube_vertices


def get_sorted_frame_files(sequence_dir):
    """Get sorted list of frame files in a sequence"""
    bin_files = glob.glob(os.path.join(sequence_dir, "*.bin"))
    # Extract frame numbers and sort
    frame_files = sorted(bin_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    return frame_files


parser = argparse.ArgumentParser(description="Visualize point cloud with 3D bounding boxes")
parser.add_argument("--show_boxes", type=str, help="show 3D bounding boxes")
parser.add_argument("--sequence", type=str, default="01", help="sequence number to visualize")
parser.add_argument("--delay", type=float, default=0.5, help="delay between frames in seconds")
parser.add_argument("--frame", type=str, help="")
args = parser.parse_args()

sequence_dir = f"carla_data/sequences/{args.sequence}/velodyne_semantic/"
label_path = f"carla_data/sequences/{args.sequence}/labels.json"
ego_path = f"carla_data/sequences/{args.sequence}/ego.csv"
calib_path = f"carla_data/sequences/{args.sequence}/calib.json"

# Load label data once
with open(label_path, "r") as f:
    labels = json.load(f)
all_frame_objects = {str(frame['frame_id']): frame['objects'] for frame in labels}

# Get sorted frame files
frame_files = get_sorted_frame_files(sequence_dir)

if not frame_files:
    print(f"No point cloud files found in {sequence_dir}")

with open(calib_path, "r") as f:
    calib = json.load(f)
lidar_to_ego = np.array(calib["sensors"]["velodyne"]["extrinsic"]["matrix"])

lidar_2_world_map = dict()
with open(ego_path) as csvfile:
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

def get_label_color(label):
    """
    根据CARLA语义标签返回对应的RGB颜色
    参数:
        label (int): CARLA语义标签值
    返回:
        tuple: (R, G, B) 颜色值，范围0-255
    """
    color_map = {
        0: (0, 0, 0),        # Unlabeled
        1: (128, 64, 128),   # Roads
        2: (244, 35, 232),   # SideWalks
        3: (70, 70, 70),     # Building
        4: (102, 102, 156),  # Wall
        5: (190, 153, 153),  # Fence
        6: (153, 153, 153), # Pole
        7: (250, 170, 30),  # TrafficLight
        8: (220, 220, 0),   # TrafficSign
        9: (107, 142, 35),  # Vegetation
        10: (152, 251, 152),# Terrain
        11: (70, 130, 180), # Sky
        12: (220, 20, 60),  # Pedestrian
        13: (255, 0, 0),    # Rider
        14: (0, 0, 142),    # Car
        15: (0, 0, 70),     # Truck
        16: (0, 60, 100),   # Bus
        17: (0, 80, 100),   # Train
        18: (0, 0, 230),    # Motorcycle
        19: (119, 11, 32),  # Bicycle
        20: (110, 190, 160),# Static
        21: (170, 120, 50), # Dynamic
        22: (55, 90, 80),   # Other
        23: (45, 60, 150),  # Water
        24: (157, 234, 50), # RoadLine
        25: (81, 0, 81),    # Ground
        26: (150, 100, 100),# Bridge
        27: (230, 150, 140),# RailTrack
        28: (180, 165, 180) # GuardRail
    }
    # 默认返回白色(用于未知标签)
    return color_map.get(label, (255, 255, 255))

def compute_rotation_matrix(rotation):
    """
    Compute the rotation matrix from yaw, pitch, and roll angles.

    Args:
        rotation (array-like): A 3-element array representing the rotation angles (yaw, pitch, roll) in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    roll, pitch, yaw = rotation

    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    r_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    r_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    r = r_z @ r_y @ r_x

    return r

def create_3d_box(center, size, rotation):
    """
    根据中心点和长宽高创建一个 3D Bounding Box 的线框
    :param center: 3D Bounding Box 的中心点，格式为 [x, y, z]
    :param size: 3D Bounding Box 的长宽高，格式为 [length, width, height]
    :param rotation_matrix: 3x3 旋转矩阵
    :return: open3d.geometry.LineSet 对象
    """
    rotation_matrix = compute_rotation_matrix(rotation)

    center = [center[0], center[1], center[2] + size[2]/2]

    # 计算半长宽高
    half_size = np.array(size)

    # 定义 8 个顶点的相对坐标
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ]) * half_size / 2  # 缩放

    # 应用旋转矩阵
    vertices = np.dot(vertices, rotation_matrix.T)

    center = [center[0], -center[1], center[2]]

    # 将顶点平移到中心点
    vertices += np.array(center)

    # 定义 12 条边的连接关系
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面
    ])

    # 创建 LineSet 对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set

def visualize_frame(frame_id):

    frame_file = f"carla_data/sequences/{args.sequence}/velodyne_semantic/{frame_id}.bin"

    # Point cloud data type
    point_dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('cos_angle', np.float32), ('obj_idx', np.uint32), ('obj_tag', np.uint32)
    ])
    data = np.fromfile(frame_file, dtype=point_dtype)
    
    pcd = o3d.geometry.PointCloud()    

    # 设置点坐标（y已经取反）
    points = np.column_stack([data['x'], -data['y'], data['z']])
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置点颜色（根据语义标签）
    colors = np.array([get_label_color(tag) for tag in data['obj_tag']]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    boxes = []

    for obj in all_frame_objects[frame_id]:
        lines = np.array([
            [0, 1], [0, 2], [1, 3], [2, 3],  # 底面
            [4, 5], [4, 6], [5, 7], [6, 7],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ])

        verts = calculate_cube_vertices(
            transform=obj["location"],
            rotation=obj["rotation"],
            dimension=obj["dimensions"]
        )

        world_2_lidar = np.linalg.inv(lidar_2_world_map[frame_id])
        bbox_vert_lidar = [point_transform_3d(vert, world_2_lidar) for vert in verts]
        bbox_vert_lidar = np.array(bbox_vert_lidar)
        
        # 创建LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_vert_lidar)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        boxes.append(line_set)


    o3d.visualization.draw_geometries(
        [pcd] + boxes,
        window_name="LiDAR Point Cloud with Semantics",
        width=1024,
        height=768,
    )


def visualize_sequence():
    
    # Set up visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Point Cloud with Semantics", width=1024, height=768)

    # Point cloud data type
    point_dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('cos_angle', np.float32), ('obj_idx', np.uint32), ('obj_tag', np.uint32)
    ])

    pcd = o3d.geometry.PointCloud()    
    is_first_frame = True

    # Main visualization loop
    for frame_file in frame_files:

        frame_id = os.path.basename(frame_file).split('.')[0]

        # Load point cloud data
        data = np.fromfile(frame_file, dtype=point_dtype)
        points = np.column_stack([data['x'], -data['y'], data['z']])
        colors = np.array([get_label_color(tag) for tag in data['obj_tag']]) / 255.0

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        vis.clear_geometries()
        
        for obj in all_frame_objects[frame_id]:
            lines = np.array([
                [0, 1], [0, 2], [1, 3], [2, 3],  # 底面
                [4, 5], [4, 6], [5, 7], [6, 7],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
            ])

            verts = calculate_cube_vertices(
                transform=obj["location"],
                rotation=obj["rotation"],
                dimension=obj["dimensions"]
            )

            world_2_lidar = np.linalg.inv(lidar_2_world_map[frame_id])
            bbox_vert_lidar = [point_transform_3d(vert, world_2_lidar) for vert in verts]
            bbox_vert_lidar = np.array(bbox_vert_lidar)
            
            # 创建LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbox_vert_lidar)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            vis.add_geometry(line_set)
        
        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            # vis.update_geometry(pcd)
            vis.add_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(args.delay)
    

if __name__ == "__main__":
    if args.frame:
        visualize_frame(args.frame)
    else:
        visualize_sequence()