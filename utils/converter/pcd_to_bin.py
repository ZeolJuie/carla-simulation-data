import numpy as np
import open3d as o3d
import os
import argparse

def pcd_to_bin(pcd_path, bin_path):
    """
    将PCD文件转换为BIN文件（添加第五维ring index）
    :param pcd_path: 输入PCD文件路径
    :param bin_path: 输出BIN文件路径
    """
    # 读取PCD文件
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # 获取点云数据
    points = np.asarray(pcd.points)
    
    # 如果有颜色信息，可以一并处理
    if pcd.has_colors():
        colors = np.asarray(pcd.colors) * 255  # 将颜色值从[0,1]转换到[0,255]
        points = np.hstack((points, colors))
    else:
        # 如果没有颜色信息，添加强度通道(默认0)
        intensity = np.zeros((points.shape[0], 1))
        points = np.hstack((points, intensity))
    
    # 添加第五维ring index（全部设为0）
    ring_index = np.zeros((points.shape[0], 1))
    points = np.hstack((points, ring_index))
    
    # 确保数据是float32类型
    points = points.astype(np.float32)
    
    # 保存为bin文件
    points.tofile(bin_path)
    print(f"Successfully converted {pcd_path} to {bin_path} (with 5th dimension)")

def batch_convert_pcd_to_bin(pcd_dir, bin_dir):
    """
    批量转换PCD文件夹中的所有文件到BIN格式（添加第五维）
    :param pcd_dir: 包含PCD文件的目录
    :param bin_dir: 输出BIN文件的目录
    """
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    
    pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.pcd')]
    
    for pcd_file in pcd_files:
        pcd_path = os.path.join(pcd_dir, pcd_file)
        bin_path = os.path.join(bin_dir, pcd_file.replace('.pcd', '.bin'))
        pcd_to_bin(pcd_path, bin_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PCD files to BIN format with 5th dimension')
    parser.add_argument('--pcd_path', type=str, help='Input PCD file or directory')
    parser.add_argument('--bin_path', type=str, help='Output BIN file or directory')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.pcd_path):
        batch_convert_pcd_to_bin(args.pcd_path, args.bin_path)
    else:
        pcd_to_bin(args.pcd_path, args.bin_path)