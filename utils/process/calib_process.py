# 将传感器的参数记录为更适合处理的内参外参(相对于自车)
import os
import json
import sys
sys.path.append('.')

import numpy as np

from utils.geometry_utils import get_extrinsic_matrix, build_projection_matrix
import config


root_path = "carla_data/sequences"
sequence = "16"

calib_json = {
    "ego_pedestrian": "",
    "coordinate_system": "right_handed",
    "sensors": {}
}

camera_configs = {cam_cfg['name']: cam_cfg for cam_cfg in config.camera_configs}

for cam in camera_configs.keys():
    cam_calib = {
        "id": cam,
        "type": "camera",
        "extrinsic": {
            "translation": camera_configs[cam]['transforms']['location'],
            "rotation": camera_configs[cam]['transforms']['rotation'],
            "matrix": get_extrinsic_matrix(
                x=camera_configs[cam]['transforms']['location']['x'],
                y=camera_configs[cam]['transforms']['location']['x'],
                z=camera_configs[cam]['transforms']['location']['x'],
                roll=np.deg2rad(camera_configs[cam]['transforms']['rotation']['roll']),
                pitch=np.deg2rad(camera_configs[cam]['transforms']['rotation']['pitch']),
                yaw=np.deg2rad(camera_configs[cam]['transforms']['rotation']['yaw']),
            ).tolist()
        },
        "intrinsic": {
            "width": config.CAMERA_IMAGE_SIZE_X,
            "height": config.CAMERA_IMAGE_SIZE_Y,
            "fov": camera_configs[cam]['fov'],
            "matrix": build_projection_matrix(
                w=config.CAMERA_IMAGE_SIZE_X,
                h=config.CAMERA_IMAGE_SIZE_Y,
                fov=camera_configs[cam]['fov']
            ).tolist()
        }
    }
    calib_json["sensors"][cam] = cam_calib

# depth/semantic camera ~ FRONT_CAM

# lidar/semantic lidar/radar
lidars = ['velodyne', 'semantic_velodyne']
for lidar in lidars:
    lidar_calib = {
        "id": lidar,
        "type": 'lidar',
        "extrinsic": {
            "translation": {
                'x': config.LIDAR_TRANSFORM_X,
                'y': config.LIDAR_TRANSFORM_Y,
                'z': config.LIDAR_TRANSFORM_Z,
            },
            "rotation": {
                'roll': 0,
                'pitch': 0,
                'yaw': 0
            },
            "matrix": get_extrinsic_matrix(
                x=config.LIDAR_TRANSFORM_X,
                y=config.LIDAR_TRANSFORM_Y,
                z=config.LIDAR_TRANSFORM_Z,
                roll=0,
                pitch=0,
                yaw=0,
            ).tolist()
        },
        "intrinsic": {
            "channels": config.LIDAR_CHANNELS,
            "rotation_frequency": config.LIDAR_ROTATION_FREQUENCY,
            "points_per_second": config.LIDAR_POINTS_PER_SECOND,
            "range": config.LIDAR_RANGE,
            "upper_fov": config.LIDAR_UPPER_FOV,
            "lower_fov": config.LIDAR_LOWER_FOV,
        }
    }
    calib_json["sensors"][lidar] = lidar_calib


# imu/gnss
calib_json["sensors"]['imu'] = {
    "id": 'imu',
    "type": 'imu',
    "extrinsic": {
        "translation": {
            'x': config.SENSOR_TRANSFORM_X,
            'y': config.SENSOR_TRANSFORM_Y,
            'z': config.SENSOR_TRANSFORM_Z,
        },
        "rotation": {
            'roll': 0,
            'pitch': 0,
            'yaw': 0
        },
        "matrix": get_extrinsic_matrix(
            x=config.SENSOR_TRANSFORM_X,
            y=config.SENSOR_TRANSFORM_Y,
            z=config.SENSOR_TRANSFORM_Z,
            roll=0,
            pitch=0,
            yaw=0,
        ).tolist()
    },
    "intrinsic": {
        "noise_accel_stddev_x": "0.01",
        "noise_accel_stddev_y": "0.01",
        "noise_accel_stddev_z": "0.01",
        "noise_gyro_stddev_x": "0.001",
        "noise_gyro_stddev_y": "0.001",
        "noise_gyro_stddev_z": "0.001"
    }
}

calib_json["sensors"]['gnss'] = {
    "id": 'imu',
    "type": 'imu',
    "extrinsic": {
        "translation": {
            'x': config.SENSOR_TRANSFORM_X,
            'y': config.SENSOR_TRANSFORM_Y,
            'z': config.SENSOR_TRANSFORM_Z,
        },
        "rotation": {
            'roll': 0,
            'pitch': 0,
            'yaw': 0
        },
        "matrix": get_extrinsic_matrix(
            x=config.SENSOR_TRANSFORM_X,
            y=config.SENSOR_TRANSFORM_Y,
            z=config.SENSOR_TRANSFORM_Z,
            roll=0,
            pitch=0,
            yaw=0,
        ).tolist()
    },
    "intrinsic": {
        "noise_alt_stddev": "0.2",
        "noise_lat_stddev": "0.000001",
        "noise_lon_stddev": "0.000001"
    }
}

with open(os.path.join(root_path, sequence, "calib.json"), 'w') as f:
    json.dump(calib_json, f)