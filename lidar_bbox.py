"""
  visualize lidar points and 3D bounding box
"""

import open3d as o3d
import numpy as np
import json

def load_point_cloud(ply_path):
    """
    加载点云数据（PLY 文件）。
    :param ply_path: PLY 文件路径。
    :return: Open3D 点云对象。
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    return pcd


def load_bbox_data(json_path):
    """
    加载 JSON 文件中的 3D Bounding Box 数据。
    :param json_path: JSON 文件路径。
    :return: 包含 3D Bounding Box 数据的字典。
    """
    with open(json_path, 'r') as f:
        bbox_data = json.load(f)
    return bbox_data


def compute_rotation_matrix(rotation):
    """
    Compute the rotation matrix from yaw, pitch, and roll angles.

    Args:
        rotation (array-like): A 3-element array representing the rotation angles (yaw, pitch, roll) in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    roll, pitch, yaw = rotation

    print(np.rad2deg(yaw))

    yaw = -yaw

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

    center = [center[0], center[1], center[2]]

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
    ]) * half_size  # 缩放

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

def visualize_point_cloud_with_bbox(ply_path, json_path):
    """
    可视化点云和 3D Bounding Box。
    :param ply_path: PLY 文件路径。
    :param json_path: JSON 文件路径。
    """
    # 加载点云
    pcd = load_point_cloud(ply_path)

    # 加载 Bounding Box 数据
    bbox_data = load_bbox_data(json_path)

    # 创建 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 3

    # 添加点云
    vis.add_geometry(pcd)

    # 添加每个 Bounding Box
    for obj_id, bbox_info in bbox_data.items():
        bbox_center = bbox_info['bbox_center']
        bbox_extent = bbox_info['bbox_extent']
        bbox_rotation = bbox_info['rotation']
        lineset = create_3d_box(bbox_center, bbox_extent, bbox_rotation)
        vis.add_geometry(lineset)

    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":

    # 点云文件路径
    ply_path = './carla_data/sequences/01/velodyne'  # 替换为你的 PLY 文件路径
    # JSON 文件路径
    json_path = './carla_data/sequences/01/velodyne'  # 替换为你的 JSON 文件路径

    # 可视化点云和 3D Bounding Box
    visualize_point_cloud_with_bbox(ply_path, json_path)