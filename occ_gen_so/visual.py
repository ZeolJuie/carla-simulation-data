import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np

import mayavi.mlab as mlab 

from carla_sync.globals import get_global

LABEL_COLORS = get_global('LABEL_COLORS') * 255
alpha = np.ones((LABEL_COLORS.shape[0], 1)) * 255
LABEL_COLORS = np.concatenate((LABEL_COLORS, alpha), axis=1)
LABEL_COLORS = LABEL_COLORS.astype(np.uint8)
 
# LABEL_COLORS = np.array(
#     [
#         [0, 120, 10, 255],
#         [255, 120, 50, 255],  # barrier              orangey
#         [255, 192, 203, 255],  # bicycle              pink
#         [255, 255, 0, 255],  # bus                  yellow
#         [0, 150, 245, 255],  # car                  blue
#         [0, 255, 255, 255],  # construction_vehicle cyan
#         [200, 180, 0, 255],  # motorcycle           dark orange
#         [255, 0, 0, 255],  # pedestrian           red
#         [255, 240, 150, 255],  # traffic_cone         light yellow
#         [135, 60, 0, 255],  # trailer              brown
#         [160, 32, 240, 255],  # truck                purple
#         [255, 0, 255, 255],  # driveable_surface    dark pink
#         # [175,   0,  75, 255],       # other_flat           dark red
#         [139, 137, 137, 255],
#         [75, 0, 75, 255],  # sidewalk             dard purple
#         [150, 240, 80, 255],  # terrain              light green
#         [230, 230, 250, 255],  # manmade              white
#         [0, 175, 0, 255],  # vegetation           green
#         [0, 255, 127, 255],  # ego car              dark cyan
#         [255, 99, 71, 255],
#         [0, 191, 255, 255]
#     ]
# ).astype(np.uint8)

if __name__=="__main__":
    #mlab.options.offscreen = True
    from argparse import ArgumentParser
    parse = ArgumentParser()
    
    parse.add_argument('--visual_path', type=str, default="/mnt/ws-data/data/project/occ_gen/carla_sync/template_record/occ/sample_173044451416/labels.npy")
    # parse.add_argument('--visual_path', type=str, default="/mnt/ws-data/data/data/nuscenes/mini/occ_gt_no_possion/occ_gt_raw/dense_voxels_with_semantic/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin.npy")
    parse.add_argument('--visual_save_dir', type=str, default="/mnt/ws-data/data/project/SurroundOcc/tools/generate_occupancy_with_own_data/occ_result/visual_save")
    args = parse.parse_args()
    visual_path = args.visual_path
    visual_save_dir = args.visual_save_dir
    
    voxel_size = 0.4
    pc_range = [-40, -40, -3, 40, 40, 3.4]

    fov_voxels = np.load(visual_path).astype(np.float32)
 
    fov_voxels = fov_voxels[fov_voxels[..., 3] >= 0]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]


    figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
 
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS


    # mlab.savefig(os.path.join(visual_save_dir, 'occ_visual.png'))
    mlab.show()
