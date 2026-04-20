# Sampling Pose And Projection Reference

## Purpose

This document defines how to interpret the sampling pose files, how to reconstruct sensor extrinsics per frame, and how to perform projection between LiDAR, world, and camera image coordinates.

The goal is to provide a stable reference for downstream export pipelines such as nuScenes-style data conversion.

## Sampling Files

### `sampling/calibration.json`

This file stores sensor-level static metadata:

- sensor name
- sensor type
- nominal sensor relative pose from the rig specification
- camera intrinsics for camera sensors

For each sensor:

- `location`: `[x, y, z]`
- `rotation`: `[pitch, yaw, roll]` in degrees
- `camera_intrinsic`: `3x3` matrix for cameras only

This file represents the static rig definition, not the per-frame world pose.

### `sampling/ego.csv`

This file stores the ego actor world pose per sampled frame.

Columns:

- `location_x`, `location_y`, `location_z`
- `rotation_roll`, `rotation_pitch`, `rotation_yaw`

Units:

- position in meters
- rotation in degrees

This file is the primary source of the ego world transform for each sampled frame.

### `sampling/sensor_pose.csv`

This file stores the per-frame sensor relative pose with motion offsets already applied.

Important fields:

- `relative_location_x`
- `relative_location_y`
- `relative_location_z`
- `relative_rotation_pitch`
- `relative_rotation_yaw`
- `relative_rotation_roll`

Units:

- position in meters
- rotation in degrees

The relative pose may differ from the static rig pose because of motion effects such as gait bobbing or pitch modulation.

The world pose columns may be unavailable or unreliable depending on the exporter version. Downstream code should therefore reconstruct world sensor poses from:

- ego world pose from `ego.csv`
- sensor relative pose from `sensor_pose.csv`
- static fallback from `calibration.json` when a frame-specific sensor pose row is missing

### `sampling/labels.json`

This file stores per-frame 3D object annotations in world coordinates.

Each object entry contains:

- `location`: box center in world coordinates
- `dimensions`: full box size along local axes
- `rotation`: `[roll, pitch, yaw]` in radians

This rotation convention is different from the pose files and must not be mixed with them.

## Coordinate Frames

The following frames are used:

- `W`: world frame
- `E`: ego frame
- `S`: sensor-local CARLA frame
- `C_cv`: camera projection frame used by the pinhole model

### World / Ego / Sensor Frames

The pose files follow CARLA transform conventions:

- `x` forward
- `y` right
- `z` up

The transform represented by a pose is:

- rotation defined by roll, pitch, yaw
- translation defined by location

The homogeneous transform from frame `A` to frame `B` is denoted as `T_BA`.

## Transform Construction

### CARLA Pose To Homogeneous Matrix

Given:

- location `[x, y, z]`
- roll, pitch, yaw in degrees

Build a `4x4` homogeneous transform using the same rotation convention as CARLA.

The resulting matrix maps points from the local frame to the parent frame.

For example:

- ego pose in `ego.csv` defines `T_WE`
- sensor relative pose defines `T_ES`

### Sensor World Pose

For a given frame:

`T_WS = T_WE @ T_ES`

where:

- `T_WE` comes from `ego.csv`
- `T_ES` comes from `sensor_pose.csv`
- if the sensor pose row is missing, `T_ES` falls back to the static pose in `calibration.json`

### World To Sensor Transform

To transform world points into a sensor frame:

`T_SW = inverse(T_WS)`

For rigid transforms, the inverse is:

- `R_inv = R^T`
- `t_inv = -R^T t`

## Frame Alignment For Sensor Pose

The relative pose records in `sensor_pose.csv` are written from the runtime sensor transforms after motion offsets are applied. In practice, downstream consumers should align each sampled frame with the nearest preceding sensor-pose record when exact same-frame alignment is not guaranteed by the exporter.

Recommended rule:

- if a matching sensor-pose frame exists, prefer the previous sampled sensor-pose frame
- otherwise use the nearest preceding sensor-pose frame
- if none exists, fall back to static rig pose from `calibration.json`

This keeps camera and LiDAR extrinsics consistent with the sampled frame content when motion effects are active.

## Camera Projection

### Camera Intrinsics

For camera sensors, the intrinsic matrix is read from `calibration.json`:

`K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`

If the image is resized for display or export, the intrinsics must be scaled accordingly:

- multiply the first row by `new_width / original_width`
- multiply the second row by `new_height / original_height`

### CARLA Camera Frame To Projection Frame

CARLA local camera coordinates use:

- `x` forward
- `y` right
- `z` up

For standard image projection, convert from CARLA camera coordinates to the projection frame:

`[x_carla, y_carla, z_carla] -> [x_cv, y_cv, z_cv] = [y_carla, -z_carla, x_carla]`

This gives:

- `z_cv` as forward depth
- `x_cv` as horizontal image axis
- `y_cv` as vertical image axis

### World Point To Pixel

For a world point `p_w`:

1. transform to camera-local CARLA frame with `T_CW = inverse(T_WC)`
2. convert from CARLA camera frame to projection frame
3. apply intrinsics
4. divide by depth

Formally:

- `p_carla = T_CW * p_w`
- `p_cv = [p_carla.y, -p_carla.z, p_carla.x]`
- `u = (K * p_cv)_x / (K * p_cv)_z`
- `v = (K * p_cv)_y / (K * p_cv)_z`

Visibility checks:

- point must satisfy `z_cv > 0`
- projected pixel must lie within image bounds if clipped rendering is desired

## LiDAR Data Interpretation

### Saved LiDAR Format

LiDAR frames are saved as `float32` arrays shaped as `N x 4`:

- `x`
- `y`
- `z`
- intensity

During export, the stored `y` value is sign-flipped before writing to disk.

Therefore, when loading the saved LiDAR file for geometry use:

- `x_carla = x_saved`
- `y_carla = -y_saved`
- `z_carla = z_saved`

After restoring this sign, the LiDAR points are back in CARLA sensor-local coordinates.

### LiDAR To World

For LiDAR-local points `p_l`:

- restore the saved `y` sign
- use the LiDAR sensor world transform `T_WL`

Then:

`p_w = T_WL * p_l`

### LiDAR To Camera Image

To project LiDAR onto a camera image:

1. load LiDAR points from file
2. restore the `y` sign
3. transform LiDAR points to world using `T_WL`
4. transform world points to camera-local CARLA frame using `inverse(T_WC)`
5. convert to projection frame
6. project using `K`

## 3D Box Construction From `labels.json`

### Label Convention

Each object in `labels.json` provides:

- world-space box center
- full box dimensions
- rotation `[roll, pitch, yaw]` in radians

The box is centered at `location`.

The dimensions are full sizes, so half extents are:

`half = dimensions / 2`

### Local Box Corners

Construct the eight local corners using the half extents:

- `( +hx, +hy, -hz )`
- `( +hx, -hy, -hz )`
- `( -hx, +hy, -hz )`
- `( -hx, -hy, -hz )`
- `( +hx, +hy, +hz )`
- `( +hx, -hy, +hz )`
- `( -hx, +hy, +hz )`
- `( -hx, -hy, +hz )`

### Box Rotation

Build the rotation matrix from the label rotation values:

- roll in radians
- pitch in radians
- yaw in radians

Apply rotation to the local corners, then translate by the box center.

This yields the eight box corners in world coordinates.

## 3D Box Projection

To project a 3D box:

1. build eight world corners from `labels.json`
2. transform corners into camera-local CARLA coordinates
3. convert into the camera projection frame
4. clip edges against a positive near plane
5. project visible segments with the camera intrinsics
6. clip final 2D line segments to the image rectangle

### Near-Plane Clipping

3D box edges should be clipped in camera projection coordinates before image projection.

Recommended rule:

- discard edges whose both endpoints are behind the near plane
- if one endpoint is behind the near plane and the other is in front, intersect the segment with the near plane
- project the clipped segment

This avoids unstable long lines when a large box crosses very close to the camera center.

### Image-Boundary Clipping

After 2D projection, clip line segments to the image rectangle instead of dropping them outright.

This preserves box edges that partially exit the image.

## Rotation Order Summary

Different files use different rotation field orders.

### `ego.csv`

- separate columns
- `rotation_roll`
- `rotation_pitch`
- `rotation_yaw`
- degrees

### `sensor_pose.csv`

- separate columns
- `relative_rotation_roll`
- `relative_rotation_pitch`
- `relative_rotation_yaw`
- degrees

### `calibration.json`

- `rotation = [pitch, yaw, roll]`
- degrees

### `labels.json`

- `rotation = [roll, pitch, yaw]`
- radians

These conventions must be parsed explicitly and must not be treated as interchangeable.

## Recommended Extrinsic Reconstruction Pipeline

For each sampled frame:

1. read ego world pose from `ego.csv`
2. select the aligned sensor-pose frame
3. read each sensor relative pose from `sensor_pose.csv`
4. fall back to static `calibration.json` pose if needed
5. build `T_WS = T_WE @ T_ES`
6. use `inverse(T_WS)` for world-to-sensor transforms

This is the recommended basis for:

- camera metadata export
- LiDAR-to-camera association
- 3D box reprojection
- dataset conversion pipelines such as nuScenes-style records

## Recommended Validation Checks

Before using the reconstructed poses in a dataset exporter, validate:

- LiDAR projected onto camera images follows visible scene geometry
- dynamic actor boxes align with image content
- image resize operations also scale intrinsics
- LiDAR `y` sign is restored before geometry use
- label rotation order is parsed separately from pose-file rotation order
- near-plane clipping is applied before projecting 3D box edges
