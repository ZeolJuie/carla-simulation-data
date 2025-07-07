import numpy as np
import open3d as o3d
import cv2
import os
import argparse

from utils.geometry_utils import *


def load_radar_data(file_path):
    """加载雷达数据.npy文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    data = np.load(file_path)
    if data.shape[1] != 4:
        raise ValueError("数据格式应为Nx4数组 [depth, azimuth, altitude, velocity]")
    
    return data

def load_img_data(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"图像文件不存在: {file_path}")
    
    img_data = cv2.imread(file_path, cv2.COLOR_RGB2BGR)

    if img_data is None:
        raise ValueError(f"无法读取图像文件: {file_path}（可能格式不支持）")
    
    return img_data

def convert_to_cartesian(data):
    """将极坐标转换为笛卡尔坐标"""
    depth = data[:, 3]
    azimuth = data[:, 1]  # 水平角（弧度）
    altitude = data[:, 2] # 俯仰角（弧度）
    velocity = data[:, 0] # 径向速度（m/s）

    print(min(depth), max(depth))
    
    # 转换为笛卡尔坐标
    x = depth * np.cos(altitude) * np.cos(azimuth)
    y = depth * np.cos(altitude) * np.sin(azimuth)
    z = depth * np.sin(altitude)
    
    return np.column_stack((x, y, z)), velocity

def visualize_radar_point_cloud(points, velocities):
    """使用Open3D可视化点云"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 根据速度值设置颜色（归一化到[0,1]）
    norm_velocities = (velocities - velocities.min()) / (velocities.max() - velocities.min() + 1e-6)
    colors = np.zeros((len(norm_velocities), 3))
    
    # 使用coolwarm颜色映射：红色表示接近，蓝色表示远离
    colors[:, 0] = np.clip(1.5 - 1.5 * norm_velocities, 0, 1)  # 红色通道
    colors[:, 2] = np.clip(1.5 * norm_velocities - 0.5, 0, 1)  # 蓝色通道
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='CARLA雷达数据可视化', width=1024, height=768)
    vis.add_geometry(pcd)
    
    # 设置视图参数
    ctr = vis.get_view_control()
    ctr.set_front([0, -1, 0.5])  # 设置相机朝向
    ctr.set_up([0, 0, 1])        # 设置上方向
    ctr.set_zoom(0.3)            # 设置缩放
    
    # 添加文字说明
    print("\n控制说明:")
    print(" - 鼠标左键: 旋转视角")
    print(" - 鼠标右键: 平移视角")
    print(" - 滚轮: 缩放")
    print(" - 颜色编码: 红色→接近雷达, 蓝色→远离雷达")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def visualize_radar_point_img(img, points, K):
    """
        img: cv2读取的img图像数据
        points: radar下的points
        K: 相机内参-到 /carla_data meta中读取
    """
    point_camera = points[:, [1, 2, 0]]
    point_camera[:, 1] = - point_camera[:, 1]

    # 计算深度（距离）
    depths = np.linalg.norm(point_camera, axis=1)
    min_depth, max_depth = depths.min(), depths.max()

    # 使用相机矩阵进行三维到二维投影
    point_img = np.dot(K, point_camera.T)

    # 归一化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    point_img = point_img.T
    uv = point_img[:, :2].astype(int)

    for (u, v), depth in zip(uv, depths):
        # 归一化深度到 [0, 1]
        norm_depth = (depth - min_depth) / (max_depth - min_depth + 1e-6)
        
        # 计算颜色（近红[0,0,255] → 远蓝[255,0,0]）
        red = int(255 * (1 - norm_depth))  # 近处红色多
        blue = int(255 * norm_depth)       # 远处蓝色多
        color = (blue, 0, red)  # BGR顺序
        
        # 绘制点（大小也可随深度变化）
        radius = int(2 + 3 * (1 - norm_depth))      # 近处点更大
        cv2.circle(img, (u, v), radius, color, -1)  # -1表示实心圆
    
    cv2.imwrite('./radar_point_2D.png', img)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='CARLA雷达数据可视化工具')
    parser.add_argument('file_path', type=str, help='雷达数据.npy文件路径')
    parser.add_argument('mode', type=str, help='可视化2D/3D')
    args = parser.parse_args()
    
    try:
        # 加载数据
        radar_data = load_radar_data(args.file_path)
        
        # 坐标转换
        points, velocities = convert_to_cartesian(radar_data)
        
        # 可视化
        if (args.mode == '3D'):
            visualize_radar_point_cloud(points, velocities)
        elif (args.mode == '2D'):
            img_path = args.file_path.replace("/radar/", "/image/").replace(".npy", ".png")
            img = load_img_data(img_path)
            K = build_projection_matrix(1600, 1200, 120)
            img = visualize_radar_point_img(img, points, K)
        
        
    except Exception as e:
        print(f"错误: {e}")
        parser.print_help()

if __name__ == "__main__":
    main()