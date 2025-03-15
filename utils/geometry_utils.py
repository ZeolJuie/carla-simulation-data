import numpy as np


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
    point = np.array([loc.x, loc.y, loc.z, 1])
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


def get_image_point(loc, K, w2c):
    # 计算三维坐标的二维投影

    # 格式化输入坐标（loc 是一个 carla.Position 对象）
    point = np.array([loc.x, loc.y, loc.z, 1])

    # 转换到相机坐标系
    point_camera = np.dot(w2c, point)

    # 将坐标系从 UE4 的坐标系转换为标准坐标系（y, -z, x），同时移除第四个分量
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # 使用相机矩阵进行三维到二维投影
    point_img = np.dot(K, point_camera)

    # 归一化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


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