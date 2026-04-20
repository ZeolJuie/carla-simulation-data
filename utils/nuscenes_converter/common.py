#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SUPPORTED_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]
SUPPORTED_LIDAR = "LIDAR_TOP"
SUPPORTED_CHANNELS = SUPPORTED_CAMERAS + [SUPPORTED_LIDAR]

CATEGORY_ORDER = [
    "Pedestrian",
    "Car",
    "Bicycle",
    "Motorcycle",
    "Truck",
    "Bus",
    "Van",
    "TrafficLight",
    "TrafficSigns",
    "Static",
    "Poles",
    "Fences",
    "Dynamic",
    "Other",
    "GuardRail",
]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_csv_index(path, key="frame"):
    index = {}
    for row in load_csv_rows(path):
        index[str(row[key])] = row
    return index


def load_sensor_pose_index(path):
    index = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = str(row["frame"])
            sensor_name = str(row["sensor_name"])
            index.setdefault(frame_id, {})[sensor_name] = row
    return index


def normalize_frame_id(value):
    return str(int(str(value).strip()))


def sensor_record_index(calibration_payload):
    return {str(sensor["name"]): sensor for sensor in calibration_payload["sensors"]}


def carla_transform_matrix(location_xyz, roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    c_y = math.cos(yaw)
    s_y = math.sin(yaw)
    c_r = math.cos(roll)
    s_r = math.sin(roll)
    c_p = math.cos(pitch)
    s_p = math.sin(pitch)

    return np.array(
        [
            [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r, float(location_xyz[0])],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r, float(location_xyz[1])],
            [s_p, -c_p * s_r, c_p * c_r, float(location_xyz[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def resolve_sensor_pose_frame(frame_id, sensor_pose_by_frame):
    pose_frames = sorted(int(value) for value in sensor_pose_by_frame.keys())
    if not pose_frames:
        return None

    target = int(frame_id)
    candidate_index = None
    for index, pose_frame in enumerate(pose_frames):
        if pose_frame == target:
            candidate_index = index
            break
        if pose_frame > target:
            candidate_index = max(0, index - 1)
            break

    if candidate_index is None:
        candidate_index = len(pose_frames) - 1

    if pose_frames[candidate_index] == target and candidate_index > 0:
        return str(pose_frames[candidate_index - 1])
    return str(pose_frames[candidate_index])


def resolve_sensor_relative_pose(sensor_name, calibration_record, frame_sensor_pose):
    if frame_sensor_pose is not None and sensor_name in frame_sensor_pose:
        row = frame_sensor_pose[sensor_name]
        return {
            "location": [
                float(row["relative_location_x"]),
                float(row["relative_location_y"]),
                float(row["relative_location_z"]),
            ],
            "roll_deg": float(row["relative_rotation_roll"]),
            "pitch_deg": float(row["relative_rotation_pitch"]),
            "yaw_deg": float(row["relative_rotation_yaw"]),
        }

    pitch_deg, yaw_deg, roll_deg = calibration_record.get("rotation", [0.0, 0.0, 0.0])
    return {
        "location": [float(value) for value in calibration_record.get("location", [0.0, 0.0, 0.0])],
        "roll_deg": float(roll_deg),
        "pitch_deg": float(pitch_deg),
        "yaw_deg": float(yaw_deg),
    }


def build_sensor_world_matrix(ego_row, sensor_name, calibration_record, frame_sensor_pose):
    ego_matrix = build_ego_world_matrix(ego_row)
    relative_pose = resolve_sensor_relative_pose(sensor_name, calibration_record, frame_sensor_pose)
    relative_matrix = carla_transform_matrix(
        relative_pose["location"],
        roll_deg=relative_pose["roll_deg"],
        pitch_deg=relative_pose["pitch_deg"],
        yaw_deg=relative_pose["yaw_deg"],
    )
    return ego_matrix @ relative_matrix


def build_ego_world_matrix(ego_row):
    return carla_transform_matrix(
        [
            float(ego_row["location_x"]),
            float(ego_row["location_y"]),
            float(ego_row["location_z"]),
        ],
        roll_deg=float(ego_row["rotation_roll"]),
        pitch_deg=float(ego_row["rotation_pitch"]),
        yaw_deg=float(ego_row["rotation_yaw"]),
    )


def deterministic_token(*parts):
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:32]


def deterministic_timestamp_us(sequence_id, frame_id, sample_time_seconds):
    frame_component = int(normalize_frame_id(frame_id))
    sequence_component = int(hashlib.sha1(str(sequence_id).encode("utf-8")).hexdigest()[:6], 16)
    return sequence_component * 10**9 + int(round(float(sample_time_seconds) * 1_000_000.0)) + frame_component


def matrix_to_nuscenes_translation_rotation(matrix):
    translation_carla = matrix[:3, 3]
    rotation_carla = matrix[:3, :3]

    # CARLA world: x forward, y right, z up
    # nuScenes convention used here: x forward, y left, z up
    axis_flip = np.diag([1.0, -1.0, 1.0])
    rotation_nusc = axis_flip @ rotation_carla @ axis_flip
    translation_nusc = axis_flip @ translation_carla

    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation_nusc)
    quaternion_wxyz = [
        quaternion_xyzw[3],
        quaternion_xyzw[0],
        quaternion_xyzw[1],
        quaternion_xyzw[2],
    ]
    return translation_nusc.tolist(), quaternion_wxyz


def sensor_matrix_to_nuscenes_translation_rotation(matrix, sensor_modality):
    translation_carla = matrix[:3, 3]
    rotation_carla = matrix[:3, :3]

    # CARLA world: x forward, y right, z up
    # nuScenes-like world used here: x forward, y left, z up
    axis_flip = np.diag([1.0, -1.0, 1.0])
    rotation_nusc = axis_flip @ rotation_carla @ axis_flip
    translation_nusc = axis_flip @ translation_carla

    if sensor_modality == "camera":
        # `rotation_nusc` above has already converted the local frame from
        # CARLA (x forward, y right, z up) into a right-handed nuScenes-like
        # local basis (x forward, y left, z up) via the axis flip on both sides.
        # The docs define the projection frame on raw CARLA camera coords as:
        #   [x_carla, y_carla, z_carla] -> [y_carla, -z_carla, x_carla]
        # Re-expressed after the local y-axis flip, this becomes:
        #   [x_forward, y_left, z_up] -> [x_right, y_down, z_forward]
        #                                = [-y_left, -z_up, x_forward]
        camera_to_optical = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        rotation_nusc = rotation_nusc @ camera_to_optical.T

    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation_nusc)
    quaternion_wxyz = [
        quaternion_xyzw[3],
        quaternion_xyzw[0],
        quaternion_xyzw[1],
        quaternion_xyzw[2],
    ]
    return translation_nusc.tolist(), quaternion_wxyz


def rotation_matrix_to_quaternion_xyzw(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
        return [x, y, z, w]

    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
        return [x, y, z, w]

    if rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
        return [x, y, z, w]

    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    w = (rotation[1, 0] - rotation[0, 1]) / s
    x = (rotation[0, 2] + rotation[2, 0]) / s
    y = (rotation[1, 2] + rotation[2, 1]) / s
    z = 0.25 * s
    return [x, y, z, w]


def object_rotation_to_nuscenes_quaternion(rotation_rpy_rad):
    roll = float(rotation_rpy_rad[0])
    pitch = float(rotation_rpy_rad[1])
    yaw = float(rotation_rpy_rad[2])

    c_y = math.cos(yaw * 0.5)
    s_y = math.sin(yaw * 0.5)
    c_p = math.cos(pitch * 0.5)
    s_p = math.sin(pitch * 0.5)
    c_r = math.cos(roll * 0.5)
    s_r = math.sin(roll * 0.5)

    x = s_r * c_p * c_y - c_r * s_p * s_y
    y = c_r * s_p * c_y + s_r * c_p * s_y
    z = c_r * c_p * s_y - s_r * s_p * c_y
    w = c_r * c_p * c_y + s_r * s_p * s_y

    # CARLA world to nuScenes-like world axis flip.
    q_flip = [0.0, 1.0, 0.0, 0.0]
    q_out = quaternion_multiply(quaternion_multiply(q_flip, [x, y, z, w]), q_flip)
    return [q_out[3], q_out[0], q_out[1], q_out[2]]


def quaternion_multiply(q1_xyzw, q2_xyzw):
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def category_names_from_labels(label_frames):
    seen = {obj["class"] for frame in label_frames for obj in frame["objects"]}
    ordered = [name for name in CATEGORY_ORDER if name in seen]
    for extra in sorted(seen - set(CATEGORY_ORDER)):
        ordered.append(extra)
    return ordered
