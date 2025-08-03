import os
import sys
import csv
import json
import time
import argparse

import cv2
import carla
import numpy as np

sys.path.append('.')
import config
from utils.geometry_utils import calculate_cube_vertices, build_projection_matrix, get_image_point


# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Play the images of the specified sequence and display the bounding box")
parser.add_argument("sequence", type=str, help="sequenceid")
parser.add_argument("--save_frame", type=str, default=None, help="Frame ID to save after drawing bounding boxes")
args = parser.parse_args()

# 图片路径和标签路径
sequence_id = args.sequence  # 从命令行参数获取序列号
image_folder = f"./carla_data/sequences/{sequence_id}/image/CAM_FRONT"
label_path = f"./carla_data/sequences/{sequence_id}/labels.json"

# 帧率（每秒播放的帧数）
fps = 2
frame_delay = int(1000 / fps)  # 每帧的延迟时间（毫秒）
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# 加载ego数据
csv_path = f'./carla_data/sequences/{sequence_id}/ego.csv'
with open(csv_path) as csvfile:
    reader = csv.DictReader(csvfile)
    ego_dict = {
        row["frame"]: [
            row['location_x'], 
            row['location_y'], 
            row['location_z'],
            row['rotation_roll'], 
            row['rotation_pitch'], 
            row['rotation_yaw'],
        ]
        for row in reader
    }   

# 加载标签数据
with open(label_path, "r") as f:
    labels = json.load(f)

# 将标签数据按帧ID整理为字典，方便快速查找
label_dict = {str(item["frame_id"]): item["objects"] for item in labels}

# 获取图片文件列表
image_files = sorted(os.listdir(image_folder))

# 计算ego坐标系到相机坐标系的转换矩阵
front_camera_config = config.camera_configs[0]
camera_location = carla.Location(
    front_camera_config['transforms']['location']['x'],
    front_camera_config['transforms']['location']['y'],
    front_camera_config['transforms']['location']['z'],
)
camera_rotation = carla.Rotation(
    roll=front_camera_config['transforms']['rotation']['roll'],
    pitch=front_camera_config['transforms']['rotation']['pitch'],
    yaw=front_camera_config['transforms']['rotation']['yaw'],
)
camera_transform = carla.Transform(camera_location, camera_rotation)
ego_to_camera = np.array(camera_transform.get_inverse_matrix())

# 计算相机的内参
camera_K = build_projection_matrix(1600, 900, 120)


# 遍历图片文件并按帧率播放
for image_file in image_files:
    # 获取帧ID（假设图片文件名是帧ID，例如 "000001.png"）
    frame_id = os.path.splitext(image_file)[0]

    if args.save_frame is not None and frame_id != args.save_frame:
        continue

    # 加载图片
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)

    # 如果图片加载失败，跳过
    if frame is None:
        print(f"无法加载图片: {image_path}")
        continue

    # 获取当前帧的标签数据
    if frame_id in label_dict:

        # 计算当前帧全局坐标系转相机坐标系的变换矩阵
        location = carla.Location(
            float(ego_dict[frame_id][0]),
            float(ego_dict[frame_id][1]),
            float(ego_dict[frame_id][2]),
        )
        rotation = carla.Rotation(
            roll=float(ego_dict[frame_id][3]),
            pitch=float(ego_dict[frame_id][4]),
            yaw=float(ego_dict[frame_id][5]),
        )
        ego_transform = carla.Transform(location, rotation)
        world_to_ego = np.array(ego_transform.get_inverse_matrix())
        # world_to_ego = np.linalg.inv(ego_matrix)

        world_to_camera = ego_to_camera @ world_to_ego

        objects = label_dict[frame_id]

        for obj in objects:
            if obj["occlusion"] == 4:
                continue
            # 提取边界框 - 计算全局坐标系下的8个顶点坐标
            bbox_location = carla.Location(
                obj["location"][0],
                obj["location"][1],
                obj["location"][2],
            )

            bbox_rotation = carla.Rotation(
                roll=np.degrees(obj["rotation"][0]),
                pitch=np.degrees(obj["rotation"][1]),
                yaw=np.degrees(obj["rotation"][2]),
            )
            bbox_extent = carla.Vector3D(
                obj["dimensions"][0] / 2,
                obj["dimensions"][1] / 2,
                obj["dimensions"][2] / 2
            )
            bbox_transform = carla.Transform()

            bbox = carla.BoundingBox()
            bbox.location = bbox_location
            bbox.extent = bbox_extent
            
            forward_vec = ego_transform.get_forward_vector()
            ray = bbox_location - location

            if forward_vec.dot(ray) > 0:

                verts = calculate_cube_vertices(
                    transform=obj["location"],
                    rotation=obj["rotation"],
                    dimension=obj["dimensions"]
                )

                x_max = -10000
                x_min = 10000
                y_max = -10000
                y_min = 10000

                for vert in verts:
                    p = get_image_point(vert, camera_K, world_to_camera)
                    if p[0] > x_max:
                        x_max = p[0]
                    # Find the leftmost vertex
                    if p[0] < x_min:
                        x_min = p[0]
                    # Find the highest vertex
                    if p[1] > y_max:
                        y_max = p[1]
                    # Find the lowest  vertex
                    if p[1] < y_min:
                        y_min = p[1]


                class_name = obj["class"]
                occlusion = obj["occlusion"]
                object_id = str(obj["object_id"])

                if not (isinstance(x_min, (int, float)) and isinstance(y_min, (int, float))):
                    print("Invalid coordinates!")
                    continue
                try:
                    cv2.line(frame, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                    cv2.line(frame, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                    cv2.line(frame, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                    cv2.line(frame, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                except Exception:
                    print(x_max, x_min, y_max, y_min)
                    continue

                # 在边界框上方显示类别名称
                cv2.putText(frame, class_name, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 在边界框上方显示跟踪ID
                cv2.putText(frame, object_id, (int(x_max) - 20, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 如果指定了保存帧，并且当前帧是目标帧，则保存图片
    if args.save_frame is not None and frame_id == args.save_frame:
        output_path = os.path.join('visualize', f"{frame_id}_with_bbox.png")
        cv2.imwrite(output_path, frame)
        print(f"save image with bounding box: {output_path}")
        break

    # 显示图片
    cv2.imshow("Frame", frame)

    # 按帧率延迟
    if cv2.waitKey(frame_delay) & 0xFF == ord("q"): 
        break

# 释放资源
cv2.destroyAllWindows()