import os

import numpy as np


voxel_size = 0.2
pc_range = [-20, -20, -2.2, 20, 20, 1.0]
occ_size = [200, 200, 16]
camera_configs = [
    {
        'name': 'CAM_FRONT',
        'transforms': {
            'location': {'x': 0.0, 'y': 0, 'z': -0.3},
            'rotation': {'pitch': 0, 'yaw': 0, 'roll': 0}
        },
        'fov': 120,
    }

]


mapping_carla_to_nuscenes = {
    0: 10,   # Unlabeled -> Free
    1: 0,    # Roads -> Road
    2: 1,    # SideWalks -> Sidewalk
    3: 2,    # Building -> Building
    4: 2,    # Wall -> Building
    5: 8,    # Fence -> Obstacle
    6: 3,    # Pole -> Pole
    7: 3,    # TrafficLight -> Traffic Element
    8: 3,    # TrafficSign -> Traffic Element
    9: 4,    # Vegetation -> Vegetation
    10: 4,   # Terrain -> Vegetation
    11: 10,  # Sky -> Free
    12: 5,   # Pedestrian -> Human
    13: 5,   # Rider -> Human
    14: 7,   # Car -> Vehicle
    15: 7,   # Truck -> Vehicle
    16: 7,   # Bus -> Vehicle
    17: 7,   # Train -> Vehicle
    18: 6,   # Motorcycle -> Vehicle
    19: 6,   # Bicycle -> Vehicle
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

def world_to_camera(point, cam_pose):

    """将点从世界坐标系转换到相机坐标系"""

    # 提取旋转和平移
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]

    # 转换到相机坐标系
    point_cam = R.T @ (point - t)
    return point_cam
 

def bresenham_3d(start, end, pc_range=None, voxel_size=None):

    """
    严格修正的3D Bresenham算法：
    1. 动态选择主步进轴（X/Y/Z中跨度最大的方向）
    2. 处理所有边界条件（包括起点=终点的情况）
    3. 返回np.ndarray类型的体素坐标数组
    参数:
    start: [x,y,z] 起点坐标（世界坐标系）
    end: [x,y,z] 终点坐标（世界坐标系）
    pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    voxel_size: 体素大小（单位：米）
    返回:
    np.ndarray: 射线穿过的体素坐标 [[x_idx, y_idx, z_idx], ...]
    """

    # === 1. 参数校验 ===
    assert pc_range is not None and voxel_size is not None, "必须提供pc_range和voxel_size"

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    # === 2. 转换为体素网格坐标 ===
    grid_min = np.array(pc_range[:3])
    start_voxel = ((start - grid_min) / voxel_size).astype(int)
    end_voxel = ((end - grid_min) / voxel_size).astype(int)

    # === 3. 初始化Bresenham算法参数 ===
    x0, y0, z0 = start_voxel
    x1, y1, z1 = end_voxel
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    sz = 1 if z0 < z1 else -1
    voxels = []

    # === 4. 动态选择主步进轴 ===
    if dx >= dy and dx >= dz:
        # X轴为主步进轴
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx

        for _ in range(dx + 1):
            voxels.append([x0, y0, z0])
            if x0 == x1:
                break

            if err1 > 0:
                y0 += sy
                err1 -= 2 * dx

            if err2 > 0:
                z0 += sz
                err2 -= 2 * dx

            err1 += 2 * dy
            err2 += 2 * dz
            x0 += sx

    elif dy >= dx and dy >= dz:

        # Y轴为主步进轴
        err1 = 2 * dx - dy
        err2 = 2 * dz - dy
        for _ in range(dy + 1):
            voxels.append([x0, y0, z0])
            if y0 == y1:
                break

            if err1 > 0:
                x0 += sx
                err1 -= 2 * dy

            if err2 > 0:
                z0 += sz
                err2 -= 2 * dy

            err1 += 2 * dx
            err2 += 2 * dz
            y0 += sy

    else:

        # Z轴为主步进轴
        err1 = 2 * dy - dz
        err2 = 2 * dx - dz
        for _ in range(dz + 1):
            voxels.append([x0, y0, z0])
            if z0 == z1:
                break

            if err1 > 0:
                y0 += sy
                err1 -= 2 * dz

            if err2 > 0:
                x0 += sx
                err2 -= 2 * dz

            err1 += 2 * dy
            err2 += 2 * dx
            z0 += sz

    # === 5. 返回结果 ===
    return np.array(voxels, dtype=int)

def compute_camera_visibility_mask(
    occupied_voxels, camera_poses, camera_configs,
    pc_range, voxel_size, grid_shape
):

    """改进的相机可见性计算"""

    mask = np.zeros(grid_shape, dtype=np.int8)
    occupied_voxels_set = {tuple(v) for v in occupied_voxels}

    for cam_pose, cam_config in zip(camera_poses, camera_configs):

        cam_center = cam_pose[:3, 3]

        for voxel in occupied_voxels:
            voxel_center = (voxel + 0.5) * voxel_size + pc_range[:3]

            # 进行射线追踪
            ray_voxels = bresenham_3d(cam_center, voxel_center, pc_range, voxel_size)

            for v in ray_voxels:
                v_tuple = tuple(v)

                # 检查体素是否在网格范围内
                if not (0 <= v[0] < grid_shape[0] and
                        0 <= v[1] < grid_shape[1] and
                        0 <= v[2] < grid_shape[2]):
                    continue

                # 标记为可见
                mask[v_tuple] = 1

                # 如果是被占用体素，终止射线
                if v_tuple in occupied_voxels_set:
                    break

    return mask

def get_extrinsic_matrix(x, y, z, roll, pitch, yaw):
    """
    根据平移和欧拉角生成4x4外参矩阵
    
    参数:
        x, y, z: 平移分量 (单位: 米)
        roll, pitch, yaw: 绕X/Y/Z轴的旋转角度 (单位: 弧度)
        (roll: X轴, pitch: Y轴, yaw: Z轴)
    
    返回:
        4x4 numpy数组表示的齐次变换矩阵
        格式:
        [[R11, R12, R13, x],
         [R21, R22, R23, y],
         [R31, R32, R33, z],
         [0,   0,   0,   1]]
    """

    # 计算旋转矩阵的各元素 (使用简写符号)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # 构建旋转矩阵 (Z-Y-X顺序，即yaw->pitch->roll)
    rotation_matrix = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr]
    ])
    
    # 构建4x4齐次变换矩阵
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = [x, y, z]
    
    return extrinsic


camera_poses = []
for cam_cfg in camera_configs:
    cam_pose = get_extrinsic_matrix(
        x=cam_cfg['transforms']['location']['x'],
        y=-cam_cfg['transforms']['location']['y'],
        z=cam_cfg['transforms']['location']['z'],
        roll=np.deg2rad(cam_cfg['transforms']['rotation']['roll']),
        pitch=np.deg2rad(cam_cfg['transforms']['rotation']['pitch']),
        yaw=np.deg2rad(-cam_cfg['transforms']['rotation']['yaw']-90)
    )
camera_poses.append(cam_pose)

sequences = [os.path.join("/home/zmh/codes/carla-simulation-data/nuscenes_carla/gts", f"scene-{i+1:04d}") for i in range(0, 100)]

# sequences = [os.path.join("/home/zmh/codes/FlashOCC/data/nuscenes/gts", f"scene-{i+1:04d}") for i in range(0, 10)]

for s in sequences:
    print(f"------ start process sequence {s} -------")
    samples = os.listdir(s)
    samples_file_path = [os.path.join(s, sample, 'labels.npz')for sample in samples]
    for idx, occ_file in enumerate(samples_file_path):
        labels = np.load(occ_file, allow_pickle=True)
        semantics = labels['semantics']
        mask_camera = labels['mask_camera']
        mask_lidar = labels['mask_lidar']
        occupied_voxels = np.argwhere(semantics != 0)
        
        mask_camera = compute_camera_visibility_mask(
            occupied_voxels, camera_poses, camera_configs, pc_range, voxel_size, occ_size
        )

        # 保留所有的pedestrian, vehicle等小物体
        obj_occ_class = [5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        obj_occ_index = np.isin(semantics, obj_occ_class).astype(int)
        
        mask_camera = obj_occ_index | mask_camera

        dict_npz = {"semantics": semantics,  "mask_camera": mask_camera, "mask_lidar": mask_lidar}
        np.savez(os.path.join(occ_file), **dict_npz)
        print(f"--- {occ_file} --- process end! ({ idx+1 } / { len(samples_file_path) })")