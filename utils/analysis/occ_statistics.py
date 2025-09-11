import os
import pickle
import random

import numpy as np

occ_data_root = '/home/zmh/codes/carla-simulation-data/nuscenes_carla/gts_40'
# occ_data_root = '/home/zmh/codes/FlashOCC/data/nuscenes/gts'

scenes = os.listdir(occ_data_root)
random.seed(42)  # 设置随机种子以确保可重复性
scenes = random.sample(scenes, min(40, len(scenes)))


total_counts = np.zeros(30)

for scene in scenes:

    print(f"=========== {scene} ===========")
    scene_path = os.path.join(occ_data_root, scene)
    samples = os.listdir(scene_path)

    frame_counts = np.zeros(30)

    for sample in samples:
        occ_labels = os.path.join(scene_path, sample, 'labels.npz')
        with np.load(occ_labels) as data:
            semantics = data['semantics']
            unique_values, counts = np.unique(semantics, return_counts=True)
            frame_counts[unique_values] += counts

    total = np.sum(frame_counts)

    # 计算每个元素的占比（百分比）
    percentages = (frame_counts / total) * 100

    for i, value in enumerate(percentages):
        if value == 0:
            print(f"{i}\t{value:.6f}\t0.000000%")
        else:
            print(f"{i}\t{value:.6f}\t{value:.6f}%")

    total_counts += frame_counts


mapping_carla_to_nuscenes = {
    0: 10,   # Unlabeled -> Free
    1: 0,    # Roads -> Road
    2: 1,    # SideWalks -> Sidewalk
    3: 2,    # Building -> Building
    4: 2,    # Wall -> Building
    5: 8,    # Fence -> Obstacle
    6: 3,    # Pole -> Pole
    7: 4,    # TrafficLight -> Traffic Element
    8: 4,    # TrafficSign -> Traffic Element
    9: 5,    # Vegetation -> Vegetation
    10: 5,   # Terrain -> Vegetation
    11: 10,  # Sky -> Free
    12: 6,   # Pedestrian -> Human
    13: 6,   # Rider -> Human
    14: 7,   # Car -> Vehicle
    15: 7,   # Truck -> Vehicle
    16: 7,   # Bus -> Vehicle
    17: 7,   # Train -> Vehicle
    18: 7,   # Motorcycle -> Vehicle
    19: 7,   # Bicycle -> Vehicle
    20: 8,   # Static -> Obstacle
    21: 8,   # Dynamic -> Obstacle
    22: 8,   # Other -> Obstacle
    23: 10,  # Water -> Void
    24: 0,   # RoadLine -> Road
    25: 9,   # Ground -> Ground
    26: 2,   # Bridge -> Building
    27: 10,  # RailTrack -> Void
    28: 3    # GuardRail -> Pole
}

nusc_class_frequencies = np.zeros(11)
for carla_idx in mapping_carla_to_nuscenes.keys():
    nusc_class_frequencies[mapping_carla_to_nuscenes[carla_idx]] += total_counts[carla_idx]

print(nusc_class_frequencies)

print(total_counts)
total = np.sum(total_counts)

print(total)
percentages = (total_counts / total) * 100
for i, value in enumerate(percentages):
    if value == 0:
        print(f"{i}\t{value:.6f}\t0.000000%")
    else:
        print(f"{i}\t{value:.6f}\t{value:.6f}%")