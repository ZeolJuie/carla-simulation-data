import open3d as o3d
import numpy as np

import argparse
import json
import time


parser = argparse.ArgumentParser(description="Visualize point cloud with 3D bounding boxes")
parser.add_argument("--show_boxes", type=str, help="show 3D bounding boxes")
args = parser.parse_args()


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

def visualize_with_open3d(bin_file):

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # 加载数据
    point_dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('cos_angle', np.float32), ('obj_idx', np.uint32), ('obj_tag', np.uint32)
    ])
    data = np.fromfile(bin_file, dtype=point_dtype)

    # data = data[data['obj_idx'] == 1311]

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置点坐标（y已经取反）
    points = np.column_stack([data['x'], data['y'], data['z']])
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置点颜色（根据语义标签）
    colors = np.array([get_label_color(tag) for tag in data['obj_tag']]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 添加点云到可视化窗口
    # vis.add_geometry(pcd)

    if args.show_boxes:
        # 获取当前帧的标签数据
        label_path = f"./carla_data/sequences/07/labels.json"

        # 加载标签数据
        with open(label_path, "r") as f:
            labels = json.load(f)
        
        all_frame_objects = {str(frame['frame_id']): frame['objects'] for frame in labels}

        frame_id = '004340'
        for obj in all_frame_objects[frame_id]:
            # 提取 3D 边界框数据
            dimensions = obj["dimensions"]  # [height, width, length]
            location = obj["location"]      # [x, y, z]
            rotation = obj["rotation"]      # 欧拉角

            # 创建 3D 边界框
            bbox = create_3d_box(location, dimensions, rotation)
            vis.add_geometry(bbox)
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="LiDAR Point Cloud with Semantics",
        width=1024,
        height=768,
    )


# 调用示例
visualize_with_open3d("carla_data/sequences/04/velodyne_semantic/003040.bin")