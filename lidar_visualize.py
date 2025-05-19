import open3d as o3d
import numpy as np
import os
import json
import time
import argparse

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


# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Visualize point cloud with 3D bounding boxes")
parser.add_argument("sequence", type=str, help="sequenceid")
parser.add_argument("--frame_id", type=str, default=None, help="Frame ID to display (e.g., '000001')")
parser.add_argument("--save_image", type=str, default=None, help="Path to save the image (e.g., 'frame_000001.png')")
args = parser.parse_args()

# 点云路径和标签路径
sequence_id = args.sequence  # 从命令行参数获取序列号
lidar_folder = f"./carla_data/sequences/{sequence_id}/velodyne"
label_path = f"./carla_data/sequences/{sequence_id}/labels.json"

# 加载标签数据
with open(label_path, "r") as f:
    labels = json.load(f)

# 将标签数据按帧ID整理为字典，方便快速查找
label_dict = {str(item["frame_id"]): item["objects"] for item in labels}

# 获取点云文件列表
ply_files = sorted(os.listdir(lidar_folder))

# 创建 Open3D 可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

vis.get_render_option().point_size = 1

# 如果指定了 frame_id，则只显示该帧
if args.frame_id:
    ply_file = f"{args.frame_id}.ply"
    if ply_file not in ply_files:
        print(f"Frame {args.frame_id} not found.")
        exit()

    # 加载点云
    ply_path = os.path.join(lidar_folder, ply_file)
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 如果点云加载失败，跳过
    if not point_cloud.has_points():
        print(f"load point cloud failed: {ply_path}")
        exit()

    # 清空可视化窗口
    vis.clear_geometries()

    # 添加点云到可视化窗口
    vis.add_geometry(point_cloud)

    # 获取当前帧的标签数据
    if args.frame_id in label_dict:
        objects = label_dict[args.frame_id]
        for obj in objects:
            # 提取 3D 边界框数据
            dimensions = obj["dimensions"]  # [height, width, length]
            location = obj["location"]  # [x, y, z]
            rotation = obj["rotation"]  # 欧拉角

            # 创建 3D 边界框
            bbox = create_3d_box(location, dimensions, rotation)
            vis.add_geometry(bbox)

    # 更新可视化窗口
    vis.poll_events()
    vis.update_renderer()

    # 如果指定了保存图片的路径，则保存当前帧为图片
    if args.save_image:
        vis.capture_screen_image(args.save_image)
        print(f"Saved image to {args.save_image}")

    # 保持窗口打开，直到用户关闭
    while True:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

else:
    # 如果没有指定 frame_id，则按帧率播放所有帧
    for ply_file in ply_files:
        # 获取帧ID（假设点云文件名是帧ID，例如 "000001.ply"）
        frame_id = os.path.splitext(ply_file)[0]

        # 加载点云
        ply_path = os.path.join(lidar_folder, ply_file)
        point_cloud = o3d.io.read_point_cloud(ply_path)

        # 如果点云加载失败，跳过
        if not point_cloud.has_points():
            print(f"load point cloud failed: {ply_path}")
            continue

        # 清空可视化窗口
        vis.clear_geometries()

        # 添加点云到可视化窗口
        vis.add_geometry(point_cloud)

        # 获取当前帧的标签数据
        if frame_id in label_dict:
            objects = label_dict[frame_id]
            for obj in objects:
                # 提取 3D 边界框数据
                dimensions = obj["dimensions"]  # [height, width, length]
                location = obj["location"]  # [x, y, z]
                rotation = obj["rotation"]  # 欧拉角

                # 创建 3D 边界框
                bbox = create_3d_box(location, dimensions, rotation)
                vis.add_geometry(bbox)

        # 更新可视化窗口
        vis.poll_events()
        vis.update_renderer()

        # 按帧率延迟
        time.sleep(1.0 / 20)

# 关闭可视化窗口
vis.destroy_window()