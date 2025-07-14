import os
import sys
import csv
from typing import List, Any

import numpy as np
import carla

sys.path.append('.')
from utils.geometry_utils import build_projection_matrix

# 定义根目录路径
source_dir = "/home/zhoumohan/codes/carla-simulation-data/carla_data"
dst_dir = "/home/zhoumohan/codes/carla-simulation-data/carla_kitti"

# time.txt


# id timestamp


# calib.txt



# P0 P1 P2 P3... Tr
# P: 相机内参
# Tr: velodyne坐标系转换到左边相机系统坐标

# poses.txt
# cam -> world


def convert_to_kitti_format(sequences: List[str]):
    
    # create root directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # create subdir for each sequence
    for seq_id in sequences:

        seq_dir = os.path.join(dst_dir, "sequences", seq_id)
        os.makedirs(seq_dir, exist_ok=True)
        
        # 1. process image, create soft link for image folder
        source_image_dir = os.path.join(source_dir, "sequences", seq_id, "image/CAM_FRONT")
        target_image_dir = os.path.join(seq_dir, "image_2")
        
        if os.path.exists(target_image_dir):
            print(f"Warning: Target directory already exists: '{target_image_dir}'")
            
        try:
            os.symlink(source_image_dir, target_image_dir, target_is_directory=True)
        except OSError as e:
            print(f"Failed to create symlink: {e}")


        # 2. process calibration, P2 Tr(Lidar->camera)
        p2_matrix = build_projection_matrix(1600, 900, 120)
        p2_line = "P2: " + " ".join([f"{x:.12f}" for x in p2_matrix.flatten()])
        
        lidar_transform = carla.Transform(
            location = carla.Location(0, 0, 1.5),
            rotation = carla.Rotation(0, 0, 0)
        )
        lidar2world = np.array(lidar_transform.get_matrix())

        camera_transform = carla.Transform(
            location = carla.Location(0.2, 0, 0.7),
            rotation = carla.Rotation(0, 0, 0)
        )
        world2camera = np.array(camera_transform.get_inverse_matrix())

        tr = world2camera @ lidar2world
        tr_line = "P2: " + " ".join([f"{x:.12f}" for x in tr.flatten()[0:12]])

        # Write to file
        with open(os.path.join(dst_dir, "sequences", seq_id, "calib.txt"), "w") as f:
            f.write(p2_line + "\n" + tr_line + '\n')

        
        # process timestamp
        ego_csv_path = os.path.join(source_dir, "sequences", seq_id, "ego.csv")
        
        with open(ego_csv_path) as csvfile:
            egos = csv.DictReader(csvfile)
            start_timestamp = next(egos)["timestamp"]
            times_line = str(format(0, ".6e")) + '\n' 
            for ego in egos:
                timestamp_ref = float(ego["timestamp"]) - float(start_timestamp)
                timestamp_ref_line = str(format(timestamp_ref, ".6e")) + '\n'
                times_line += timestamp_ref_line
        with open(os.path.join(dst_dir, "sequences", seq_id, "times.txt"), "w") as f:
            f.write(times_line)

        print('done')
        





if __name__ == '__main__':

    sequence = ['01', '02', '03']

    convert_to_kitti_format(sequence)
