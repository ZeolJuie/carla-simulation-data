import numpy as np
import carla


def point_is_occluded(point, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    x, y, vertex_depth = map(int, point)

    from itertools import product
    neigbours = product((1, -1), repeat=2)

    is_occluded = []
    for dy, dx in neigbours:
        # If the point is on the boundary
        if x == (depth_map.shape[1] - 1) or y == (depth_map.shape[0] - 1):
            is_occluded.append(True)
        # If the depth map says the pixel is closer to the camera than the actual vertex
        elif depth_map[y + dy, x + dx] < vertex_depth:
            is_occluded.append(True)
        else:
            is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)


def calculate_occlusion_stats(vertices_pos2d, depth_image, MAX_RENDER_DEPTH_IN_METERS):
    """ 
        筛选bbox八个顶点中实际可见的点 
        vertices_pos2d: points_image
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    image_h, image_w = depth_image.shape

    for x_2d, y_2d, vertex_depth in vertices_pos2d:

        # 点在可见范围中，并且没有超出图片范围
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((x_2d, y_2d), image_h, image_w):

            is_occluded = point_is_occluded(
                (x_2d, y_2d, vertex_depth), depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
        构建一个 相机内参矩阵(Intrinsic Matrix), 用于将3D空间中的点投影到2D图像平面上
        input:
            w: width of image
            h: height of image
            fov: Field of View eg:120/90
            is_behind_camera: 于指示是否处理相机后方的点, 如果为True, 则焦距f取负值

        output:
            K = [   fx  0   cx
                    0   fy  cy
                    0   0   1   ]
            
            generally: fx = fy = f
            f = w / (2.0 * np.tan(fov * np.pi / 360.0))

            cx, cy: 图像的主点, 通常是图像中心

    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    """
        Calculate 2D projection of 3D coordinate
        :loc - a carla.Position object
        :K - projection matrix  (3D point -> 2D point)
            将相机坐标系下的3D点投影到2D图像平面 (内参)
        :w2c - world to camera matrix
            描述相机的位置和姿态    (外参)
            将世界坐标系下的3D点 转换到 相机坐标系下的3D点
            [   R(3x3)  T(3X1)
                0(1x3)  1     ]
    """
    # Format the input coordinate (loc is a carla.Position object)
    if isinstance(loc, carla.Location):
        point = np.array([loc.x, loc.y, loc.z, 1])
    elif isinstance(loc, list):
        point = np.array(loc.append(1))
    elif isinstance(loc, np.ndarray):
        point = np.concatenate([loc, [1]])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为旋转矩阵。
    :param roll: 绕 X 轴的旋转角度（弧度）。
    :param pitch: 绕 Y 轴的旋转角度（弧度）。
    :param yaw: 绕 Z 轴的旋转角度（弧度）。
    :return: 3x3 旋转矩阵。
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx  # 旋转顺序：Z-Y-X
    return R


def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角(roll, pitch, yaw)。
    :param R: 3x3 旋转矩阵。
    :return: 欧拉角(roll, pitch, yaw) (弧度)。
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def get_center_point(verts):
    """
    :param verts: list<Location> 顶点列表
    :return 顶点构成的Bounding Box的中心点坐标
    """
    # 计算中心点
    center_x = sum(v.x for v in verts) / len(verts)
    center_y = sum(v.y for v in verts) / len(verts)
    center_z = sum(v.z for v in verts) / len(verts)
    return center_x, center_y, center_z

def calculate_cube_vertices(transform, rotation, dimension):
    """
    计算立方体的8个顶点坐标
    
    参数:
        transform (list/tuple/np.array): 立方体中心的平移向量 [x, y, z]
        rotation (list/tuple/np.array): 立方体的旋转角度 [rx, ry, rz] (弧度)
        dimension (list/tuple/np.array): 立方体的尺寸 [width, height, depth]
    
    返回:
        np.array: 8个顶点坐标的数组，形状为(8, 3)
    """
    # 转换为numpy数组
    transform = np.array(transform)
    rotation = np.array(rotation)
    dimension = np.array(dimension)
    
    # 立方体的局部坐标 (未旋转和平移前)
    half_dim = dimension / 2.0
    vertices_local = np.array([
        [-1, -1, -1],  # 0: 左前下
        [-1,  1, -1],  # 1: 右前下
        [ 1, -1, -1],  # 2: 右后下
        [ 1,  1, -1],  # 3: 左后下
        [-1, -1,  1],  # 4: 左前上
        [-1,  1,  1],  # 5: 右前上
        [ 1, -1,  1],  # 6: 右后上
        [ 1,  1,  1]   # 7: 左后上
    ]) * half_dim
    
    # 创建旋转矩阵 (绕x, y, z轴旋转)
    rx, ry, rz = rotation
    rotation_matrix = euler_to_rotation_matrix(rx, ry, rz)
    
    # 应用旋转和平移
    vertices_global = np.dot(vertices_local, rotation_matrix.T) + transform
    
    return vertices_global

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
