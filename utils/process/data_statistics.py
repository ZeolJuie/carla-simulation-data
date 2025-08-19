import os
import json
import subprocess

import pandas as pd
# Basic Statistics
"""
    总帧数​​（Total Frames）：采集的总数据量
    ​​各传感器数据量​​：
    RGB 图像数量
    深度图数量
    语义分割图数量
    LiDAR 点云数量
    IMU/GPS 数据条数
    ​​数据存储大小​​（GB/TB）

    采集总时长​​（Hours/Minutes）
    ​​帧率（FPS）​​：
    摄像头帧率（如 10 FPS）
    LiDAR 扫描频率（如 20 Hz）

"""



# Scene Diversity
"""
    2. 场景多样性（Scene Diversity）​​
    ​​(1) 天气与光照条件​​
    统计不同天气场景的分布：

    晴天（ClearNoon）
    雨天（WetNoon）
    雾天（Foggy）
    夜晚（Night）
    ​​(2) 地图与道路类型​​
    城市（Town01, Town02）
    乡村（Town03）
    高速公路（Town04）
    复杂交叉路口（Town05）
    ​​(3) 交通密度​​
    车辆数量（平均每帧）
    行人数量（平均每帧）
    静态障碍物（如交通灯、标志牌）
"""

# Object Detection Statistics
"""
    (1) 目标类别分布​​
    统计不同类别的目标数量（可用于检查类别不平衡问题）：

    车辆（Car, Truck, Motorcycle）
    行人（Pedestrian）
    交通标志（Traffic Sign）
    其他（Bicycle, Traffic Light）

    ​​(2) 目标尺寸分布​​
    2D 检测：Bounding Box 的宽高分布
    3D 检测：Bounding Box 的长宽高分布
    
    ​​(3) 目标运动状态​​
    静止物体占比
    运动物体速度分布
"""

# Behavior Statistics
"""
    轨迹分析​​
    平均行走距离
    转弯次数
    停止/启动次数
"""

def object_detection_statistics():
    """ 检测物体统计：各个序列的各目标数量分布、尺寸分布、运动状态分布 """
    data_root = "carla_data/sequences"

    sequences = sorted(os.listdir(data_root))

    columns = [
        'Pedestrian',
        'Car',
        'Truck',
        'Bicycle',
        'TrafficLight',
        'TrafficSigns',
        'Poles',
        'Static',
        'Size(<0.)'
    ]
    df = pd.DataFrame(columns=columns)

    for idx, sequence in enumerate(sequences):

        df.loc[idx, 'sequence'] = sequence
        sequence_dir = os.path.join(data_root, sequence)
        labels_filename = os.path.join(sequence_dir, 'labels.json')

        with open(labels_filename, 'r', encoding='utf-8') as file:
            labels = json.load(file)
        
        pedestrians_num = 0
        vehicles_num = 0
        statics_num = 0
        for frame in labels:
            pass


def pedestrian_behavior_statistics():
    """
        统计各序列行人的运动轨迹
    """


def scene_diversity_statistic():
    """ 场景统计：包括各序列的场景信息（天气、地图、流量）"""

    data_root = "carla_data/sequences"

    sequences = sorted(os.listdir(data_root))

    columns = [
        'sequence',
        'weather',
        'map',
        'vehicles/frame',
        'pedestrians/frame',
        'statics/frame',
    ]
    df = pd.DataFrame(columns=columns)

    for idx, sequence in enumerate(sequences):

        df.loc[idx, 'sequence'] = sequence
        sequence_dir = os.path.join(data_root, sequence)
        labels_filename = os.path.join(sequence_dir, 'labels.json')

        with open(labels_filename, 'r', encoding='utf-8') as file:
            labels = json.load(file)
        
        pedestrians_num = 0
        vehicles_num = 0
        statics_num = 0

        for frame in labels:
            pedestrians = [obj for obj in frame["objects"] if obj["class"] == 'Pedestrian' and obj["occlusion"] == 0]
            pedestrians_num += len(pedestrians)
            vehicles = [obj for obj in frame["objects"] if obj["class"] in ['Car', 'Truck', 'Bus']]
            vehicles_num += len(vehicles)
            statics = [obj for obj in frame["objects"] if obj["class"] in ['Static']]
            statics_num += len(statics)
        
        df.loc[idx, 'vehicles/frame'] = vehicles_num / len(labels)
        df.loc[idx, 'pedestrians/frame'] = pedestrians_num / len(labels)
        df.loc[idx, 'statics/frame'] = statics_num / len(labels)

    print(df)
        

def basic_statistics():
    """
    统计各个序列的采集帧数、采集时常，各个传感器数据的磁盘占用量
    """
    data_root = "carla_data/sequences"

    sequences = sorted(os.listdir(data_root))

    columns = [
        'sequence',
        'total_frames',
        'duration_sec',
        'CAM_BACK_RIGHT', 
        'CAM_BACK', 
        'CAM_BACK_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'velodyne_semantic',
        'radar',
        'depth',
        'semantic',
        'velodyne',
    ]

    df = pd.DataFrame(columns=columns)

    for idx, sequence in enumerate(sequences):

        df.loc[idx, 'sequence'] = sequence
        sequence_dir = os.path.join(data_root, sequence)
        
        for dirpath, dirnames, filenames in os.walk(sequence_dir):
            if len(dirnames) == 0:

                sensor_name = dirpath.split('/')[-1]
                if sensor_name not in columns:
                    continue

                if pd.isna(df.loc[idx, 'total_frames']):
                    df.loc[idx, 'total_frames'] = len(filenames)
                    df.loc[idx, 'duration_sec'] = len(filenames) * 0.5
                
                total_size = 0
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                size_mb = total_size / (1024 * 1024)
                df.loc[idx, sensor_name] = size_mb
        

    # statistic full data
    columns_to_sum = df.columns.difference(['sequence', 'total_frames', 'duration_sec'])
    df['Total'] = df[columns_to_sum].sum(axis=1)

    df.loc['Total'] = df.drop('sequence', axis=1).sum()

    print(df)


def main():

    basic_statistics()

    scene_diversity_statistic()
    



if __name__ == "__main__":
    
    main()

