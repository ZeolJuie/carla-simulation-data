import sys
sys.path.append('.')

import json
import os

import cv2

from utils.geometry_utils import *


"""
    通过深度图筛选bbox八个顶点中实际可见的点
    可见6个以上 0
    可见4/5个 1
    可见4个以下 2
"""


PRELIMINARY_FILTER_DISTANCE = 50


def rgb_to_depth(depth_image):
    """
    将深度相机的 RGB 图像转换为深度值（以米为单位）。
    
    :param depth_image: 深度相机的 RGB 图像（H x W x 3）。
    :return: 深度值数组（H x W）。
    """
    # 将 RGB 通道转换为深度值
    depth_meters = np.dot(depth_image[:, :, :3], [65536.0, 256.0, 1.0]) / (256 * 256 * 256 - 1)
    depth_meters *= 1000  # 转换为米
        
    return depth_meters


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


def center_to_vertices(center, size, rotation):

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
    ]) * half_size / 2 # 缩放

    # 应用旋转矩阵
    vertices = np.dot(vertices, rotation_matrix.T)

    # 将顶点平移到中心点
    vertices += np.array(center)

    return vertices

def get_image_point(loc, K):

    # loc还是lidar坐标系下的

    point = np.column_stack([
        loc[:, 1],   # 新第1列 = 原第2列取负
        -loc[:, 2],  # 新第2列 = 原第1列
        loc[:, 0]    # 新第3列 = 原第3列
    ])

    # now project 3D->2D using the camera matrix
    point_img = (K @ point.T).T

    # normalize
    point_img[:, 0] /= point_img[:, 2]
    point_img[:, 1] /= point_img[:, 2]

    return point_img


def process_single_frame(depth_path, labels, K):
    """处理单个深度图像并更新遮挡信息"""
    # 读取深度图
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"警告：无法读取深度图 {depth_path}")
        return
    
    depth_map = rgb_to_depth(depth_image)
    frame_id = os.path.splitext(os.path.basename(depth_path))[0]
    
    # 查找对应的标签数据
    frame_labels = next((item for item in labels if str(item["frame_id"]) == frame_id), None)
    if frame_labels is None:
        print(f"警告：未找到帧 {frame_id} 的标签数据")
        return
    
    # 处理每个对象
    for obj in frame_labels["objects"]:
        vertices = center_to_vertices(obj["location"], obj["dimensions"], obj["rotation"])
        points_image = get_image_point(vertices, K)
        
        num_visible_vertices, _ = calculate_occlusion_stats(points_image, depth_map, PRELIMINARY_FILTER_DISTANCE)
        
        if num_visible_vertices >= 6:
            obj["occlusion"] = 0
        elif num_visible_vertices >= 4:
            obj["occlusion"] = 1
        elif num_visible_vertices >= 2:
            obj["occlusion"] = 2
        else:
            obj["occlusion"] = 3
    
    print(f"{depth_path} processed completed")

# 主处理函数
def process_all_frame_occlusion(sequence_dir):
    """处理序列目录下的所有帧的遮挡信息"""
    # 读取标签文件
    label_path = os.path.join(sequence_dir, "labels.json")
    with open(label_path, "r") as f:
        labels = json.load(f)
    
    # 计算内参矩阵
    K = build_projection_matrix(1600, 1200, 120)
    
    # 处理所有帧对应的标注信息和深度图
    depth_dir = os.path.join(sequence_dir, "depth")
    for depth_file in os.listdir(depth_dir):
        if depth_file.endswith(".png"):
            depth_path = os.path.join(depth_dir, depth_file)
            process_single_frame(depth_path, labels, K)

    # 保存更新后的标签
    with open(label_path, "w") as f:
        json.dump(labels, f, indent=4)
    print(f"已更新 {len(labels)} 帧的遮挡信息")


def process_all_frame_labels():

    label_path = os.path.join(sequence_dir, "labels.json")
    with open(label_path, "r") as f:
        labels = json.load(f)
    
    for frame in labels:
        for obj in frame["objects"]:
            new_class = obj["class"]
            if "pedestrian" in new_class:
                new_class = "Pedestrian"
            if "vehicle" in new_class or "car" in new_class:
                new_class = "Car"
            obj["class"] = new_class

    # 保存更新后的标签
    with open(label_path, "w") as f:
        json.dump(labels, f, indent=4)

def process_occlusion_by_lidar():
    """
        使用（语义）激光雷达判定遮挡情况
    """
    # 1. 读取语义激光雷达数据和label.json

    # 2. 根据雷达第五维instance id, 确定动态障碍物（Car， Pedestrian）的点云覆盖情况




def generate_static_object_id(sequence_dir):
    """
        CityObjectLabel获取的静态物体没有ID属性, 由于物体的全局translation固定, 根据全局坐标区分instance并分配sequence内唯一的Object ID
    """

    # 1. 读取label.json
    label_path = os.path.join(sequence_dir, "labels.json")
    with open(label_path, "r") as f:
        labels = json.load(f)

    # 2. 建立map： key-Tuple(x, y, z), value-object ID
    object_map = dict()

    # 创建待分配的new id，在该序列中唯一，从10000开始分配 CityObject id
    new_id = 10000

    # 3. 遍历frame，当前帧下，统计已分配的ID
    for frame in labels:
        exist_ids = [obj['object_id'] for obj in frame['objects']]
        
        # 4. 遍历当前帧下所有ID = -1 （采集时标记的CityObject）的Object
        for obj in frame['objects']:
            if obj['object_id'] != -1:
                continue

            object_location = tuple(obj['location'])

            # 如果其translation已经在key中，修改ID
            if object_location in object_map.keys():
                obj['object_id'] = object_map[object_location]

            # 如果还没有在key中（新进入场景采集范围的object），则创建分配ID，并记录map
            else:
                # 确保new id未被分配
                while new_id in exist_ids:
                    new_id += 1
                obj['object_id'] = new_id
                exist_ids.append(new_id)
                object_map[object_location] = obj['object_id']

    # 保存更新后的标签
    with open(label_path, "w") as f:
        json.dump(labels, f, indent=4)
        print("Generate static bbject ID end, new label file is saved.")
    
# 运行处理
if __name__ == "__main__":
    sequence_dir = "carla_data/sequences/01"
    
    # 统计动态障碍物的点云覆盖数量

    # 使用深度相机信息，计算相机视角下的遮挡
    # process_occlusion_by_lidar()
    # process_all_frame_occlusion(sequence_dir)

    # 去除Poles灯泡

    # CityO物体分配instance id
    generate_static_object_id(sequence_dir)


    


