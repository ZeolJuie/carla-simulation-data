import sys
sys.path.append('.')
sys.path.append('..')
 
import cv2, os
import mayavi.mlab as mlab
import numpy as np

from carla_sync.globals import get_global

FREE_LABEL = 17
VOXEL_SIZE = 0.4
POINT_CLOUD_RANGE = [-40, -40, -3]

LABEL_COLORS = get_global('LABEL_COLORS') * 255
alpha = np.ones((LABEL_COLORS.shape[0], 1)) * 255
LABEL_COLORS = np.concatenate((LABEL_COLORS, alpha), axis=1)
LABEL_COLORS = LABEL_COLORS.astype(np.uint8)
 
 
colors_nuscenes = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],  # 4 car  Crimson
        [233, 150, 70, 255],   # 5 cons. Veh  Orangered
        [255, 61, 99, 255],  # 6 motorcycle  Darkorange
        [0, 0, 230, 255], # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone  Red
        [255, 140, 0, 255],# 9 trailer  Slategrey
        [255, 99, 71, 255],# 10 truck Burlywood
        [0, 207, 191, 255],    # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        # [255, 255, 255, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegeyation
        # [255, 255, 255, 255], # free label 
    ]
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

def occ_show(pred_occ, mask_camera = None, data_type = 'nuscenes'):
    if data_type == 'nuscenes':
        vmax = 16
        free_label = FREE_LABEL
        colors = colors_nuscenes
    elif data_type == 'carla':
        vmax = 24
        free_label = 25
        colors = LABEL_COLORS

    fov_voxels = voxel2points(pred_occ, mask_camera, free_label=free_label)
  
    fov_voxels = fov_voxels[fov_voxels[..., 3] > 0] if data_type=='carla' else fov_voxels
    figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
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

    mlab.show()
    
if __name__=="__main__":
    # pred_occ_path = "/mnt/ws-data/data/data/nuscenes/mini/gts/scene-0103/3e8750f331d7499e9b5123e9eb70f2e2/labels.npz"
    pred_occ_path = "./carla_sync/template_record/occ/sample_173044451416/labels.npz"
    pred_occ = np.load(pred_occ_path, allow_pickle=True)['semantics']
    
    occ_show(pred_occ, data_type='carla')