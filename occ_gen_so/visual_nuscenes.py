import sys
import time
import glob
from pyface.timer.api import Timer
sys.path.append('.')
sys.path.append('..')
 
import cv2, os
import mayavi.mlab as mlab
import numpy as np

from carla_sync.globals import get_global

FREE_LABEL = 17
VOXEL_SIZE = 0.2
POINT_CLOUD_RANGE = [-40, -40, -3]

LABEL_COLORS = get_global('LABEL_COLORS') * 255
alpha = np.ones((LABEL_COLORS.shape[0], 1)) * 255
LABEL_COLORS = np.concatenate((LABEL_COLORS, alpha), axis=1)
LABEL_COLORS = LABEL_COLORS.astype(np.uint8)
 
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
 
colors_nuscenes = np.array(
    list(color_map.values())
).astype(np.uint8)

def voxel2points(pred_occ, mask_camera = None, free_label = FREE_LABEL):
    x = np.linspace(0, pred_occ.shape[0] - 1, pred_occ.shape[0])
    y = np.linspace(0, pred_occ.shape[1] - 1, pred_occ.shape[1])
    z = np.linspace(0, pred_occ.shape[2] - 1, pred_occ.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z, pred_occ], axis=-1)
    if mask_camera is None:
        mask = pred_occ != free_label
    else:
        mask = np.logical_and(pred_occ != free_label, mask_camera)
    fov_voxels = vv[mask].astype(np.float32)
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * VOXEL_SIZE
    fov_voxels[:, 0] += POINT_CLOUD_RANGE[0]
    fov_voxels[:, 1] += POINT_CLOUD_RANGE[1]
    fov_voxels[:, 2] += POINT_CLOUD_RANGE[2]
    return fov_voxels

current_frame = 0

def occ_show(pred_occ, mask_camera = None, data_type = 'nuscenes'):
    if data_type == 'nuscenes':
        vmax = 16
        free_label = FREE_LABEL
        colors = colors_nuscenes
    elif data_type == 'carla':
        vmax = 24
        free_label = 25
        colors = LABEL_COLORS
        

    def update_frame():
        global current_frame

        mlab.clf()  # 清除上一帧

        pred = pred_occ[current_frame]

        fov_voxels = voxel2points(pred, mask_camera, free_label=free_label)
    
        fov_voxels = fov_voxels[fov_voxels[..., 3] > 0] if data_type=='carla' else fov_voxels
        mlab.clf()  # 清除上一帧

        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=VOXEL_SIZE - 0.05*VOXEL_SIZE,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=vmax,
        )

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        current_frame = (current_frame + 1) % len(pred_occ)

    figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    timer = Timer(500, update_frame)

    mlab.show()

if __name__=="__main__":
    # pred_occ_path = "/mnt/ws-data/data/data/nuscenes/mini/gts/scene-0103/3e8750f331d7499e9b5123e9eb70f2e2/labels.npz"
    # pred_occ_path = "/home/zhoumohan/codes/carla-simulation-data/carla_data/sequences/08/occ/sample_313330/labels.npz"
    pred_occ_path = "/home/zhoumohan/codes/carla-simulation-data/carla_data/sequences/01/occ/sample_061000/labels.npz"  # 替换为你的序列目录
    pred_occ = np.load(pred_occ_path, allow_pickle=True)['semantics']
    
    # occ_show(pred_occ, data_type='carla')
    path_pattern = "/home/zhoumohan/codes/carla-simulation-data/carla_data/sequences/01/occ/sample_*/labels.npz"

    # 获取所有匹配的文件路径
    file_paths = sorted(glob.glob(path_pattern))

    pred_occ_list = []

    for file in file_paths:
        pred_occ = np.load(file, allow_pickle=True)['semantics']
        pred_occ_list.append(pred_occ)
    occ_show(pred_occ_list, data_type='carla')