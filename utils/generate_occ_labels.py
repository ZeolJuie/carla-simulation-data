#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.postprocess_dataset.common import (  # noqa: E402
    build_sensor_world_matrix,
    load_json,
    normalize_frame_id,
    read_semantic_lidar_points_xyz,
    resolve_sensor_pose_frame,
    sensor_record_index,
    sequence_dirs_from_root,
    transform_points,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "occ" / "default.toml"
DEFAULT_SENSOR_NAME = "LIDAR_TOP_SEMANTIC"


CARLA_SEMANTIC_TO_OCC = {
    0: 10,
    1: 0,
    2: 1,
    3: 2,
    4: 2,
    5: 8,
    6: 3,
    7: 3,
    8: 3,
    9: 4,
    10: 4,
    11: 10,
    12: 5,
    13: 5,
    14: 7,
    15: 7,
    16: 7,
    17: 7,
    18: 6,
    19: 6,
    20: 8,
    21: 8,
    22: 8,
    23: 10,
    24: 0,
    25: 9,
    26: 2,
    27: 10,
    28: 3,
}


DYNAMIC_CLASS_TO_OCC = {
    "Pedestrian": 5,
    "Rider": 5,
    "Motorcycle": 6,
    "Bicycle": 6,
    "Car": 7,
    "Truck": 7,
    "Bus": 7,
    "Van": 7,
    "Train": 7,
    "Other": 8,
    "Dynamic": 8,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate occupancy labels.npz from sampled CARLA sequences using semantic LiDAR, "
            "per-frame labels.json, calibration.json, ego.csv, and sensor_pose.csv."
        )
    )
    parser.add_argument(
        "input_path",
        help="Sequence directory or dataset root, e.g. artifacts/sequences/<id> or a root containing many sequences.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to OCC generation TOML config.",
    )
    parser.add_argument(
        "--sensor-name",
        default=DEFAULT_SENSOR_NAME,
        help="Semantic LiDAR sensor name under sampling/velodyne_semantic.",
    )
    parser.add_argument(
        "--output-dirname",
        default="occ",
        help="Output directory name under each sequence root.",
    )
    parser.add_argument(
        "--len-sequence",
        type=int,
        default=None,
        help="Temporal merge window length. Overrides config if set.",
    )
    parser.add_argument(
        "--len-side",
        type=int,
        default=None,
        help="Look-back overlap between windows. Overrides config if set.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip target frames whose labels.npz already exists.",
    )
    parser.add_argument(
        "--skip-missing-lidar",
        action="store_true",
        help="Skip missing semantic LiDAR frames instead of failing.",
    )
    return parser.parse_args()


def load_occ_config(path):
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    occ_cfg = payload.get("occ", payload)
    return {
        "voxel_size": float(occ_cfg["voxel_size"]),
        "pc_range": np.asarray(occ_cfg["pc_range"], dtype=np.float64),
        "occ_size": tuple(int(value) for value in occ_cfg["occ_size"]),
        "self_range": np.asarray(occ_cfg["self_range"], dtype=np.float64),
        "len_sequence": int(occ_cfg.get("len_sequence", 40)),
        "len_side": int(occ_cfg.get("len_side", 4)),
    }


def iter_sequence_dirs(input_path):
    path = Path(input_path)
    if (path / "sampling" / "labels.json").exists():
        return [path]
    return sequence_dirs_from_root(path)


def load_csv_index_normalized(path, key="frame"):
    index = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index[normalize_frame_id(row[key])] = row
    return index


def load_sensor_pose_index_normalized(path):
    index = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = normalize_frame_id(row["frame"])
            sensor_name = str(row["sensor_name"])
            index.setdefault(frame_id, {})[sensor_name] = row
    return index


def load_label_index(labels_path):
    labels = load_json(labels_path)
    labels_by_frame = {}
    raw_frame_ids = {}
    for frame in labels:
        raw_frame_id = str(frame["frame_id"])
        frame_id = normalize_frame_id(raw_frame_id)
        labels_by_frame[frame_id] = frame.get("objects", [])
        raw_frame_ids[frame_id] = raw_frame_id
    return labels_by_frame, raw_frame_ids


def transform_points_batch(points_xyz, matrix):
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return transform_points(matrix, points_xyz)


def rotation_matrix_from_radians(roll_rad, pitch_rad, yaw_rad):
    c_y = math.cos(yaw_rad)
    s_y = math.sin(yaw_rad)
    c_r = math.cos(roll_rad)
    s_r = math.sin(roll_rad)
    c_p = math.cos(pitch_rad)
    s_p = math.sin(pitch_rad)
    return np.array(
        [
            [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
            [s_p, -c_p * s_r, c_p * c_r],
        ],
        dtype=np.float64,
    )


def build_box_in_lidar(world_to_lidar, object_record):
    center_world = np.asarray(object_record["location"], dtype=np.float64)
    center_world_h = np.concatenate([center_world, np.array([1.0], dtype=np.float64)])
    center_lidar = np.dot(world_to_lidar, center_world_h)[:3]

    roll_rad, pitch_rad, yaw_rad = [float(value) for value in object_record["rotation"]]
    object_rotation_world = rotation_matrix_from_radians(roll_rad, pitch_rad, yaw_rad)
    lidar_rotation = np.dot(world_to_lidar[:3, :3], object_rotation_world)
    yaw_lidar = math.atan2(lidar_rotation[1, 0], lidar_rotation[0, 0])

    dimensions = np.asarray(object_record["dimensions"], dtype=np.float64)
    return np.concatenate([center_lidar, dimensions, np.array([yaw_lidar], dtype=np.float64)])


def filter_dynamic_objects(objects):
    dynamic_objects = []
    for obj in objects:
        if int(obj.get("object_id", -1)) <= 0:
            continue
        if obj.get("type_id") is None and obj.get("velocity") is None and obj.get("speed") is None:
            continue
        dynamic_objects.append(obj)
    return dynamic_objects


def occ_semantic_from_dynamic_class(class_name):
    return DYNAMIC_CLASS_TO_OCC.get(str(class_name), 8)


def occ_semantic_from_static_tag(obj_tag):
    return CARLA_SEMANTIC_TO_OCC.get(int(obj_tag), 8)


def remove_ego_points(points_lidar_xyz, self_range):
    if points_lidar_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (np.abs(points_lidar_xyz[:, 0]) > self_range[0])
        | (np.abs(points_lidar_xyz[:, 1]) > self_range[1])
        | (np.abs(points_lidar_xyz[:, 2]) > self_range[2])
    )


def filter_points_in_range(points_xyz, pc_range):
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (points_xyz[:, 0] >= pc_range[0])
        & (points_xyz[:, 0] < pc_range[3])
        & (points_xyz[:, 1] >= pc_range[1])
        & (points_xyz[:, 1] < pc_range[4])
        & (points_xyz[:, 2] >= pc_range[2])
        & (points_xyz[:, 2] < pc_range[5])
    )


def aggregate_semantics_by_voxel(points_xyz, point_semantics, voxel_size, pc_range, occ_size):
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int8)

    voxels = np.floor((points_xyz - pc_range[:3]) / voxel_size).astype(np.int32)
    valid = np.all(voxels >= 0, axis=1)
    valid &= voxels[:, 0] < occ_size[0]
    valid &= voxels[:, 1] < occ_size[1]
    valid &= voxels[:, 2] < occ_size[2]
    voxels = voxels[valid]
    point_semantics = point_semantics[valid]

    if voxels.size == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int8)

    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    voxel_semantics = np.zeros((unique_voxels.shape[0],), dtype=np.int8)
    for voxel_index in range(unique_voxels.shape[0]):
        semantics = point_semantics[inverse == voxel_index]
        semantic_values, counts = np.unique(semantics, return_counts=True)
        voxel_semantics[voxel_index] = semantic_values[np.argmax(counts)]

    return unique_voxels, voxel_semantics


def build_occ_grid(voxels, semantics, occ_size):
    grid = np.zeros(occ_size, dtype=np.int8)
    if voxels.size == 0:
        return grid
    occ_voxels = voxels.copy()
    occ_voxels[:, 1] *= -1
    occ_voxels[:, 1] += occ_size[1] - 1
    grid[occ_voxels[:, 0], occ_voxels[:, 1], occ_voxels[:, 2]] = semantics
    return grid


def build_frame_inputs(sequence_dir, sensor_name):
    sampling_dir = Path(sequence_dir) / "sampling"
    labels_by_frame, raw_frame_ids = load_label_index(sampling_dir / "labels.json")
    calibration_payload = load_json(sampling_dir / "calibration.json")
    calibration_by_sensor = sensor_record_index(calibration_payload)
    ego_by_frame = load_csv_index_normalized(sampling_dir / "ego.csv")
    sensor_pose_path = sampling_dir / "sensor_pose.csv"
    sensor_pose_by_frame = (
        load_sensor_pose_index_normalized(sensor_pose_path)
        if sensor_pose_path.exists()
        else {}
    )

    if sensor_name not in calibration_by_sensor:
        raise KeyError("Sensor {} not found in {}".format(sensor_name, sampling_dir / "calibration.json"))

    lidar_dir = sampling_dir / "velodyne_semantic" / sensor_name
    lidar_paths_by_frame = {
        normalize_frame_id(path.stem): path
        for path in sorted(lidar_dir.glob("*.bin"))
    }
    frame_ids = sorted(labels_by_frame.keys(), key=lambda value: int(value))

    return {
        "sampling_dir": sampling_dir,
        "labels_by_frame": labels_by_frame,
        "raw_frame_ids": raw_frame_ids,
        "calibration_record": calibration_by_sensor[sensor_name],
        "ego_by_frame": ego_by_frame,
        "sensor_pose_by_frame": sensor_pose_by_frame,
        "lidar_paths_by_frame": lidar_paths_by_frame,
        "frame_ids": frame_ids,
    }


def compute_lidar_world_matrix(frame_id, ego_by_frame, calibration_record, sensor_pose_by_frame, sensor_name):
    ego_row = ego_by_frame.get(frame_id)
    if ego_row is None:
        raise KeyError("Frame {} missing from ego.csv".format(frame_id))
    pose_frame_id = resolve_sensor_pose_frame(frame_id, sensor_pose_by_frame)
    frame_sensor_pose = sensor_pose_by_frame.get(pose_frame_id, {}) if pose_frame_id is not None else {}
    return build_sensor_world_matrix(
        ego_row=ego_row,
        sensor_name=sensor_name,
        calibration_record=calibration_record,
        frame_sensor_pose=frame_sensor_pose,
    )


def collect_frame_record(frame_id, frame_inputs, sensor_name, self_range, skip_missing_lidar):
    lidar_path = frame_inputs["lidar_paths_by_frame"].get(frame_id)
    if lidar_path is None or not lidar_path.exists():
        if skip_missing_lidar:
            return None
        raise FileNotFoundError(
            "Missing semantic LiDAR frame for frame_id={} under {}".format(
                frame_id,
                frame_inputs["sampling_dir"] / "velodyne_semantic" / sensor_name,
            )
        )

    lidar_world = compute_lidar_world_matrix(
        frame_id=frame_id,
        ego_by_frame=frame_inputs["ego_by_frame"],
        calibration_record=frame_inputs["calibration_record"],
        sensor_pose_by_frame=frame_inputs["sensor_pose_by_frame"],
        sensor_name=sensor_name,
    )
    world_to_lidar = np.linalg.inv(lidar_world)

    points_lidar_xyz, raw = read_semantic_lidar_points_xyz(lidar_path)
    obj_ids = raw["obj_idx"].astype(np.int64, copy=False)
    obj_tags = raw["obj_tag"].astype(np.int64, copy=False)
    point_semantics = np.asarray([occ_semantic_from_static_tag(tag) for tag in obj_tags], dtype=np.int8)

    dynamic_objects = filter_dynamic_objects(frame_inputs["labels_by_frame"].get(frame_id, []))
    object_tokens = [int(obj["object_id"]) for obj in dynamic_objects]
    object_categories = np.asarray(
        [occ_semantic_from_dynamic_class(obj.get("class")) for obj in dynamic_objects],
        dtype=np.int8,
    )
    gt_boxes = (
        np.asarray([build_box_in_lidar(world_to_lidar, obj) for obj in dynamic_objects], dtype=np.float64)
        if dynamic_objects
        else np.empty((0, 7), dtype=np.float64)
    )

    static_mask = remove_ego_points(points_lidar_xyz, self_range)
    for object_token in object_tokens:
        static_mask &= obj_ids != object_token

    static_points_lidar = points_lidar_xyz[static_mask]
    static_semantics = point_semantics[static_mask]
    static_points_world = transform_points_batch(static_points_lidar, lidar_world)

    object_points_local = []
    for object_token, box in zip(object_tokens, gt_boxes):
        object_mask = obj_ids == object_token
        object_points = points_lidar_xyz[object_mask]
        if object_points.size == 0:
            object_points_local.append(np.empty((0, 3), dtype=np.float64))
            continue
        centered = object_points - box[:3]
        rotation = rotation_matrix_from_radians(0.0, 0.0, -box[6])
        canonical_points = np.dot(rotation, centered.T).T
        object_points_local.append(canonical_points)

    return {
        "frame_id": frame_id,
        "output_frame_id": lidar_path.stem,
        "world_to_lidar": world_to_lidar,
        "static_points_world": static_points_world,
        "static_semantics": static_semantics,
        "object_tokens": object_tokens,
        "object_categories": object_categories,
        "object_points_local": object_points_local,
        "gt_boxes": gt_boxes,
    }


def merge_object_canonical_points(records):
    merged_points = {}
    merged_semantics = {}
    for record in records:
        for token, semantic, canonical_points in zip(
            record["object_tokens"],
            record["object_categories"],
            record["object_points_local"],
        ):
            if canonical_points.size == 0:
                continue
            merged_points.setdefault(token, []).append(canonical_points)
            merged_semantics[token] = semantic

    return {
        token: np.concatenate(parts, axis=0)
        for token, parts in merged_points.items()
        if parts
    }, merged_semantics


def restore_dynamic_points_for_target(target_record, merged_object_points, merged_object_semantics):
    token_to_index = {
        token: index
        for index, token in enumerate(target_record["object_tokens"])
    }
    dynamic_points = []
    dynamic_semantics = []

    for token, canonical_points in merged_object_points.items():
        box_index = token_to_index.get(token)
        if box_index is None:
            continue
        box = target_record["gt_boxes"][box_index]
        rotation = rotation_matrix_from_radians(0.0, 0.0, box[6])
        points = np.dot(rotation, canonical_points.T).T + box[:3]
        dynamic_points.append(points)
        dynamic_semantics.append(
            np.full((points.shape[0],), merged_object_semantics[token], dtype=np.int8)
        )

    if not dynamic_points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int8)

    return np.concatenate(dynamic_points, axis=0), np.concatenate(dynamic_semantics, axis=0)


def write_occ_labels(output_dir, semantics_grid, occ_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "semantics": semantics_grid,
        "mask_camera": np.ones(occ_size, dtype=np.int8),
        "mask_lidar": np.ones(occ_size, dtype=np.int8),
    }
    np.savez(output_dir / "labels.npz", **payload)


def compute_sequence_occ(
    sequence_dir,
    config,
    sensor_name,
    output_dirname,
    len_sequence,
    len_side,
    skip_existing,
    skip_missing_lidar,
):
    frame_inputs = build_frame_inputs(sequence_dir, sensor_name)
    output_root = Path(sequence_dir) / output_dirname
    frame_ids = frame_inputs["frame_ids"]
    all_sample_number = len(frame_ids)

    generated = 0
    skipped_existing_count = 0
    skipped_missing_count = 0
    i = 0

    while i < all_sample_number:
        left_side = i if i == 0 else i - len_side
        right_side = min(left_side + len_sequence, all_sample_number)
        if right_side + len_sequence - len_side >= all_sample_number:
            right_side = all_sample_number

        records = []
        missing_in_window = 0
        for frame_index in range(left_side, right_side):
            frame_id = frame_ids[frame_index]
            record = collect_frame_record(
                frame_id=frame_id,
                frame_inputs=frame_inputs,
                sensor_name=sensor_name,
                self_range=config["self_range"],
                skip_missing_lidar=skip_missing_lidar,
            )
            if record is None:
                missing_in_window += 1
                continue
            records.append(record)

        if not records:
            i = right_side
            skipped_missing_count += missing_in_window
            continue

        static_world_parts = [
            record["static_points_world"]
            for record in records
            if record["static_points_world"].size > 0
        ]
        static_semantic_parts = [
            record["static_semantics"]
            for record in records
            if record["static_semantics"].size > 0
        ]
        merged_static_world = (
            np.concatenate(static_world_parts, axis=0)
            if static_world_parts
            else np.empty((0, 3), dtype=np.float64)
        )
        merged_static_semantics = (
            np.concatenate(static_semantic_parts, axis=0)
            if static_semantic_parts
            else np.empty((0,), dtype=np.int8)
        )
        merged_object_points, merged_object_semantics = merge_object_canonical_points(records)

        right_side_gen = len(records) if right_side == all_sample_number else max(0, len(records) - len_side)
        for local_index in range(0, right_side_gen):
            target_record = records[local_index]
            occ_dir = output_root / "sample_{}".format(target_record["output_frame_id"])
            occ_path = occ_dir / "labels.npz"
            if skip_existing and occ_path.exists():
                skipped_existing_count += 1
                continue

            static_points_lidar = transform_points_batch(merged_static_world, target_record["world_to_lidar"])
            static_mask = filter_points_in_range(static_points_lidar, config["pc_range"])
            static_points_lidar = static_points_lidar[static_mask]
            static_semantics = merged_static_semantics[static_mask]

            dynamic_points, dynamic_semantics = restore_dynamic_points_for_target(
                target_record=target_record,
                merged_object_points=merged_object_points,
                merged_object_semantics=merged_object_semantics,
            )
            dynamic_mask = filter_points_in_range(dynamic_points, config["pc_range"])
            dynamic_points = dynamic_points[dynamic_mask]
            dynamic_semantics = dynamic_semantics[dynamic_mask]

            if static_points_lidar.size > 0 and dynamic_points.size > 0:
                scene_points = np.concatenate([static_points_lidar, dynamic_points], axis=0)
                scene_semantics = np.concatenate([static_semantics, dynamic_semantics], axis=0)
            elif static_points_lidar.size > 0:
                scene_points = static_points_lidar
                scene_semantics = static_semantics
            else:
                scene_points = dynamic_points
                scene_semantics = dynamic_semantics

            voxels, voxel_semantics = aggregate_semantics_by_voxel(
                scene_points,
                scene_semantics,
                voxel_size=config["voxel_size"],
                pc_range=config["pc_range"],
                occ_size=config["occ_size"],
            )
            semantics_grid = build_occ_grid(voxels, voxel_semantics, config["occ_size"])
            write_occ_labels(occ_dir, semantics_grid, config["occ_size"])
            generated += 1

        i = right_side
        skipped_missing_count += missing_in_window

    return {
        "sequence": Path(sequence_dir).name,
        "generated": generated,
        "skipped_existing": skipped_existing_count,
        "skipped_missing": skipped_missing_count,
        "output_root": str(output_root),
    }


def main():
    args = parse_args()
    config = load_occ_config(args.config)
    if args.len_sequence is not None:
        config["len_sequence"] = int(args.len_sequence)
    if args.len_side is not None:
        config["len_side"] = int(args.len_side)

    sequence_dirs = iter_sequence_dirs(args.input_path)
    if not sequence_dirs:
        raise RuntimeError("No sequence directories found under {}".format(args.input_path))

    for sequence_dir in sequence_dirs:
        summary = compute_sequence_occ(
            sequence_dir=sequence_dir,
            config=config,
            sensor_name=args.sensor_name,
            output_dirname=args.output_dirname,
            len_sequence=config["len_sequence"],
            len_side=config["len_side"],
            skip_existing=args.skip_existing,
            skip_missing_lidar=args.skip_missing_lidar,
        )
        print(
            "{sequence}: generated={generated} skipped_existing={skipped_existing} "
            "skipped_missing={skipped_missing} output={output_root}".format(**summary)
        )


if __name__ == "__main__":
    main()
