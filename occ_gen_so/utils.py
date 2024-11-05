import open3d as o3d 
import numpy as np
from copy import deepcopy 
from nuscenes.utils.data_classes import LidarPointCloud 
from pyquaternion import Quaternion 

def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc

def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )

def nn_correspondance(verts1, verts2): # unuse
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
 
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

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