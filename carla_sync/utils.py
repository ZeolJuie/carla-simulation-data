import carla
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation


def add_open3d_axis(vis): # 加到车头上的坐标
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([ # 设点
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([ # 连线
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([ # 颜色
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

def quaternion2euler(quaternion, order='xyz'):
    r = Rotation.from_quat(quaternion)
    euler = r.as_euler(order, degrees=True)
    return euler


def matrix_to_quaternion(matrix, to_wxyz=False):
    r = Rotation.from_matrix(matrix)
    quat = r.as_quat()
    if to_wxyz:
        quat = np.roll(quat, 1)
    return quat

def rotation_matrix_to_euler_angles(rotation_matrix):
    beta = np.arctan2(-rotation_matrix[2,0], np.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))
    alpha = np.arctan2(rotation_matrix[2,1]/np.cos(beta), rotation_matrix[2,2]/np.cos(beta))
    gamma = np.arctan2(rotation_matrix[1,0]/np.cos(beta), rotation_matrix[0,0]/np.cos(beta))
    return np.array([gamma, beta, alpha])
   
def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def point_transform_3d(loc, M):
    """ 
        Transform a 3D point using a 4x4 matrix
    """
 
    point = np.array([loc.x, loc.y, loc.z, 1]) if isinstance(loc, carla.libcarla.Location) else loc
    point_transformed = np.dot(M, point)
    # normalize, 其实最后一位就是1.0
    point_transformed[0] /= point_transformed[3]
    point_transformed[1] /= point_transformed[3]
    point_transformed[2] /= point_transformed[3]
    return point_transformed[:3]

def point_transform_3d_batch(loc, M):
    """ 
        Transform a 3D point using a 4x4 matrix
        loc: Nx3 array point location 
        out: Nx3 array transformed point location 
    """
    point = np.concatenate((loc, np.ones((loc.shape[0], 1))), axis=1)
    point = point[:, :, np.newaxis]
    M = M[np.newaxis, :, :] 
    point_transformed = np.matmul(M, point)[:,:,0]
    # normalize, 其实最后一位就是1.0
    point_transformed[:, 0] /= point_transformed[:, 3]
    point_transformed[:, 1] /= point_transformed[:, 3]
    point_transformed[:, 2] /= point_transformed[:, 3]
    return point_transformed[:, :3]

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point) 
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera) # 这里np.dot是矩阵乘
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img

