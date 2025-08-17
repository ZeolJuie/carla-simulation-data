import os
import csv
import json
import math
import time
import uuid
import shutil
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from loguru import logger
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation


# 工具函数
def generate_token() -> str:
    """生成16位唯一token"""
    return str(uuid.uuid4().hex)[:16]


def euler_to_quaternion(pitch: float, yaw: float, roll: float) -> List[float]:
    """将欧拉角转换为四元数(x,y,z,w)格式"""
    cy = math.cos(math.radians(yaw) * 0.5)
    sy = math.sin(math.radians(yaw) * 0.5)
    cp = math.cos(math.radians(pitch) * 0.5)
    sp = math.sin(math.radians(pitch) * 0.5)
    cr = math.cos(math.radians(roll) * 0.5)
    sr = math.sin(math.radians(roll) * 0.5)
    
    return [
        sr * cp * cy - cr * sp * sy,    # x
        cr * sp * cy + sr * cp * sy,    # y
        cr * cp * sy - sr * sp * cy,    # z
        cr * cp * cy + sr * sp * sy     # w
    ]


def carla_quaternion_to_nuscenes(q: List[float]) -> List[float]:
    """将CARLA的四元数转换为nuScenes格式（可能需要坐标系转换）"""
    # 示例：CARLA使用UE4坐标系（左前上），nuScenes使用右前上
    # 这里假设无需转换，实际需要根据CARLA的坐标系调整
    return [q[3], q[0], q[1], q[2]]  # w, x, y, z


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K.tolist()

def copy_file(src, dst):
    """
    Copy a file from source path to destination path.
    
    Args:
        src (str): Source file path
        dst (str): Destination file path
    
    Returns:
        None
    """
    try:
        # Copy the file from source to destination
        shutil.copy(src, dst)
        # print(f"File {src} successfully copied to {dst}")
    except IOError as e:
        # Handle file operation errors (e.g., file not found, permission issues)
        print(f"Failed to copy file: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An error occurred: {e}")


# 主转换类
class CarlaToNuScenesConverter:
    def __init__(self, carla_root: str, nuscenes_output: str, sequence: str):
        self.carla_root = carla_root
        self.output_dir = nuscenes_output
        self.sequence = sequence
        self.token_map: Dict[str, str] = {}  # 存储生成的token映射 sensor_name: sensor_token
        
        # 初始化nuScenes数据结构
        self.nuscenes = {
            "log": [],
            "sensors": [],
            "calibrated_sensors": [],
            "ego_poses": [],
            "scenes": [],
            "samples": [],
            "sample_data": [],
            "category": [],
            "sample_annotation": [],
            "attribute": [],
            "instance": [],
        }
    
    def convert(self):
        """ excute scene data convert pipeline """

        # 0.创建基础的文件夹结构
        self._create_nuscenes_file_structure()

        # 以场景（序列）为单位转换数据 

        # 1.创建传感器，通常创建一次即可，且须读出
        self._create_sensors()
        # 2.创建传感器标定，考虑到方针采集传感器相对无抖动，通常创建一次即可，且须读出
        self._create_calibrated_sensors()

        # 3.创建新场景的ego pose，若已创建ego_pose.json，则在原文件上写入
        self._create_ego_poses()
        # 4.创建场景及场景对应的样本，若已创建，则在原文件上写入
        self._create_scenes_and_samples()

        # 5.创建样本数据，若已创建，则在原文件上写入
        self._create_sample_data()
        
        self._create_attribute()
        self._create_category_and_instance()
        self._create_sample_annotation()
        self._create_instance()

        self._save_json_files()

        
    
    def _create_nuscenes_file_structure(self):
        # 定义NuScenes标准子目录结构
        dirs = [
            "samples",          # 传感器样本数据 (图像、点云等)
            "sweeps",           # 连续扫描数据
            "maps",             # 地图数据
            "v1.0-trainval",    # 标注数据 (训练/验证)
            "v1.0-test",        # 标注数据 (测试)
            "v1.0-mini",        # 迷你版数据集
        ]
        
        # 传感器子目录 (samples和sweeps下)
        sensor_dirs = [
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "LIDAR_TOP",
            "RADAR_BACK_LEFT",
            "RADAR_BACK_RIGHT",
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
        ]

        version_json_files = [
            "attribute.json",
            "map.json",
            "visibility.json"
        ]

        
        # 确保基础路径存在
        base_path = Path(self.output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        print(f"正在创建NuScenes文件结构在: {base_path}")
        
        # 创建主目录
        for dir_name in dirs:
            dir_path = base_path / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"创建目录: {dir_path}")
            
            # 为samples和sweeps创建传感器子目录
            if dir_name in ["samples", "sweeps"]:
                for sensor in sensor_dirs:
                    sensor_path = dir_path / sensor
                    sensor_path.mkdir(exist_ok=True)
                    print(f"  创建传感器目录: {sensor_path}")
        
        for version in [
            "v1.0-trainval",
            "v1.0-test",
            "v1.0-mini",
        ]:
            version_dir = os.path.join(self.output_dir, version)
            # 创建一些必要的空文件
            for json_file in version_json_files:
                json_path = os.path.join(version_dir, json_file)
                with open(json_path, 'w') as f:
                    if json_file == 'visibility.json':
                        visibility_data = [
                            {
                                "description": "visibility of whole object is 0",
                                "token": "4",
                                "level": "v0"
                            },
                            {
                                "description": "visibility of whole object is between 0 and 40%",
                                "token": "3",
                                "level": "v0-40"
                            },
                            {
                                "description": "visibility of whole object is between 40 and 60%",
                                "token": "2",
                                "level": "v40-60"
                            },
                            {
                                "description": "visibility of whole object is between 60 and 80%",
                                "token": "1",
                                "level": "v60-80"
                            },
                            {
                                "description": "visibility of whole object is between 80 and 100%",
                                "token": "0",
                                "level": "v80-100"
                            }
                        ]
                        json.dump(visibility_data, f, indent=4)
                    if json_file == 'map.json':
                        map_data = [
                            {
                            "category": "semantic_prior",
                            "token": "53992ee3023e5494b90c316c183be829",
                            "filename": "maps/53992ee3023e5494b90c316c183be829.png",
                            "log_tokens": [
                                "0986cb758b1d43fdaa051ab23d45582b",
                                "1c9b302455ff44a9a290c372b31aa3ce",
                                "e60234ec7c324789ac7c8441a5e49731",
                                "46123a03f41e4657adc82ed9ddbe0ba2",
                                "a5bb7f9dd1884f1ea0de299caefe7ef4",
                                "bc41a49366734ebf978d6a71981537dc",
                                "f8699afb7a2247e38549e4d250b4581b",
                                "d0450edaed4a46f898403f45fa9e5f0d",
                                "f38ef5a1e9c941aabb2155768670b92a",
                                "7e25a2c8ea1f41c5b0da1e69ecfa71a2",
                                "ddc03471df3e4c9bb9663629a4097743",
                                "31e9939f05c1485b88a8f68ad2cf9fa4",
                                "783683d957054175bda1b326453a13f4",
                                "343d984344e440c7952d1e403b572b2a",
                                "92af2609d31445e5a71b2d895376fed6",
                                "47620afea3c443f6a761e885273cb531",
                                "d31dc715d1c34b99bd5afb0e3aea26ed",
                                "34d0574ea8f340179c82162c6ac069bc",
                                "d7fd2bb9696d43af901326664e42340b",
                                "b5622d4dcb0d4549b813b3ffb96fbdc9",
                                "da04ae0b72024818a6219d8dd138ea4b",
                                "6b6513e6c8384cec88775cae30b78c0e",
                                "eda311bda86f4e54857b0554639d6426",
                                "cfe71bf0b5c54aed8f56d4feca9a7f59",
                                "ee155e99938a4c2698fed50fc5b5d16a",
                                "700b800c787842ba83493d9b2775234a"
                                ]
                            }
                        ]
                        json.dump(map_data, f, indent=4)
                    pass  # just create an empty file
                print(f"Created JSON file: {json_path}")
            
           
        
        print("NuScenes文件结构创建完成!")

    def _create_log(self):
        """ each sequence relates to a logfile """
        logs = [{
            "token": generate_token(),
            "logfile": "n015-2018-07-24-11-22-45+0800",
            "walker": "n015",
            "date_captured": "2025-06-19",
            "location": "Town10"
        }]
        self.nuscenes["log"] = logs
    
    def _create_sensors(self):
        """创建sensor.json内容"""

        # 检查sensor.json是否已经创建
        if not os.path.exists(os.path.join(self.output_dir, "v1.0-trainval", "sensor.json")):

        # CARLA传感器到nuScenes的映射
            sensor_mapping = [
                {"token": generate_token(), "channel": "LIDAR_TOP", "modality": "lidar"},
                
                # 相机传感器（nuScenes标准配置）
                {"token": generate_token(), "channel": "CAM_FRONT", "modality": "camera"},
                {"token": generate_token(), "channel": "CAM_FRONT_LEFT", "modality": "camera"},
                {"token": generate_token(), "channel": "CAM_FRONT_RIGHT", "modality": "camera"},
                {"token": generate_token(), "channel": "CAM_BACK", "modality": "camera"},
                {"token": generate_token(), "channel": "CAM_BACK_LEFT", "modality": "camera"},
                {"token": generate_token(), "channel": "CAM_BACK_RIGHT", "modality": "camera"},
            ]
            self.nuscenes["sensors"] = sensor_mapping
            # 保存token映射供后续使用
            for sensor in sensor_mapping:
                self.token_map[sensor["channel"]] = sensor["token"]

        # 若已经创建，读取传感器token
        else:
            with open(os.path.join(self.output_dir, "v1.0-trainval", "sensor.json"), "r") as f:
                existing_sensors = json.load(f)
                self.nuscenes["sensors"] = existing_sensors
                for sensor in existing_sensors:
                    self.token_map[sensor["channel"]] = sensor["token"]
                    print(f"load senesor { sensor['channel'] }, token: {sensor['token']}")
    
    def _create_calibrated_sensors(self):
        """
            创建calibrated_sensor.json内容
            暂不考虑sensor -> calibration 1对多的关系
            直接使用配置参数
        """
        # 从CARLA的calib目录读取标定数据
        
        import sys
        sys.path.append('.')
        from config import camera_configs

        # 若已经创建sensor
        if os.path.exists(os.path.join(self.output_dir, "v1.0-trainval", "calibrated_sensor.json")):
            with open(os.path.join(self.output_dir, "v1.0-trainval", "calibrated_sensor.json"), "r") as f:
                calibrated_sensors = json.load(f)
                self.nuscenes["calibrated_sensors"] = calibrated_sensors
        
            return


        # TODO: 相机外参还需要 GLOBAL(x, y, z) -> CAMERA(-y, -z, -x)
        for config in camera_configs:
            # 从欧拉角计算四元数
            rotation = config['transforms']['rotation']
            
            # GLOBAL(x, y, z) -> CAMERA(-y, -z, x) <<=>>  Roll (X) 90° , Yaw (Z) 90°
            q_carla = euler_to_quaternion(rotation['pitch'], -rotation['yaw'] - 90, rotation['roll'] - 90)

            translation_x = config['transforms']['location']['x']
            translation_y = config['transforms']['location']['y']
            translation_z = config['transforms']['location']['z']

            translation = [
                translation_x,
                -translation_y,
                translation_z
            ]

            # 构建标定数据
            cam_calib = {
                "token": generate_token(),
                "sensor_token": self.token_map[config['name']],
                "translation": translation,
                "rotation": carla_quaternion_to_nuscenes(q_carla),
                "camera_intrinsic": build_projection_matrix(1600, 900, config['fov'])
            }
            self.nuscenes["calibrated_sensors"].append(cam_calib)
        
        # LIDAR_TOP calibration
        q_carla = euler_to_quaternion(0, 0, 0)
        lidar_calib = {
            "token": generate_token(),
            "sensor_token": self.token_map['LIDAR_TOP'],
            "translation": [0, -0, 1.0],
            "rotation": carla_quaternion_to_nuscenes(q_carla),
            "camera_intrinsic": []
        }
        self.nuscenes["calibrated_sensors"].append(lidar_calib)


    def _create_scenes_and_samples(self):
        """创建scene.json和sample.json内容"""

        # 创建场景（ego创建时已经验证场景是否已存在）
        scene_token = generate_token()
        # TODO： 遍历sequence
        scene = {
            "token": scene_token,
            "name": f"CARLA_Sequence_{self.sequence}",
            "description": "Go straight through the busy streets",
            "log_token": generate_token(),                              # TODO： 应与log.json关联
            "last_sample_token": "",                                    # 将在填充samples后更新
            "first_sample_token": "",  
            "nbr_samples": 0                                            # 该场景包含的样本（关键帧）数量
        }
        
        # 生成样本（关键帧）
        samples = []

        # NOTICE：考虑到仿真世界同步模式下时间戳完全对齐，先解析ego_pose,获取所有时间戳，并作为sample的时间戳
        timestamps = sorted(self.timestamp_map.keys())

        prev_token = ""
        for i, timestamp in enumerate(timestamps):
            sample_token = generate_token()
            sample = {
                "token": sample_token,
                "timestamp": timestamp,
                "scene_token": scene_token,
                "prev": prev_token if i > 0 else "",
                "next": ""
            }
            if i > 0:
                samples[-1]["next"] = sample_token
            samples.append(sample)
            prev_token = sample_token
        
        # 更新scene信息
        scene["nbr_samples"] = len(samples)
        scene["first_sample_token"] = samples[0]["token"]
        scene["last_sample_token"] = samples[-1]["token"]
        
        self.nuscenes["scenes"].append(scene)
        self.nuscenes["samples"] = samples
    
    def _create_ego_poses(self):
        """创建ego_pose.json内容"""

        # 1.读取scene.json, 判断当前场景是否已经在数据集中
        if os.path.exists(os.path.join(self.output_dir, "v1.0-trainval", "scene.json")):

            with open(os.path.join(self.output_dir, "v1.0-trainval", "scene.json"), "r") as f:
                scenes = json.load(f)
                for sence in scenes:
                    # 判断当前场景是否已经在数据集中
                    if self.sequence == (sence["name"]).split("_")[-1]:
                        raise Exception("Sequence had been converted!")

        # 从CARLA的gnss/imu数据生成
        csv_path = f'./carla_data/sequences/{self.sequence}/ego.csv'

        # 创建时间戳
        start_timestamp = int(time.time() * 1_000_000)  # us
        
        # 记录时间戳和帧的对应关系
        self.timestamp_map = dict()
        # 记录时间戳对应的ego_pose_token
        self.ego_pose_tokens = dict()
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                delta_timestamp = int(float(row["timestamp"]) * 1_000_000)

                # 每一个ego对应的时间戳
                ego_timestamp = start_timestamp + delta_timestamp

                q_carla = euler_to_quaternion(
                    float(row['rotation_pitch']), 
                    -float(row['rotation_yaw']), 
                    float(row['rotation_roll'])
                )

                # CARLA (x, y, z) -> nuScenes(x, -y, z)
                ego_pose = {
                    'token': generate_token(),
                    'timestamp': ego_timestamp,
                    'rotation': carla_quaternion_to_nuscenes(q_carla),
                    'translation': [
                        float(row['location_x']),
                        -float(row['location_y']),
                        float(row['location_z'])
                    ]
                }

                self.nuscenes["ego_poses"].append(ego_pose)
                self.timestamp_map[ego_timestamp] = row["frame"]
                self.ego_pose_tokens[ego_timestamp] = ego_pose["token"]
                
    
    def _create_sample_data(self):
        """
            创建sample_data.json内容
            支持6个相机+LiDAR
            {
                "token": "5ace90b379af485b9dcb1584b01e7212",
                "sample_token": "39586f9d59004284a7114a68825e8eec",
                "ego_pose_token": "5ace90b379af485b9dcb1584b01e7212",
                "calibrated_sensor_token": "f4d2a6c281f34a7eb8bb033d82321f79",
                "timestamp": 1532402927814384,
                "fileformat": "pcd",
                "is_key_frame": false,
                "height": 0,
                "width": 0,
                "filename": "sweeps/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927814384.pcd",
                "prev": "f0b8593e08594a3eb1152c138b312813",
                "next": "978db2bcdf584b799c13594a348576d2"
            },
        """

        logger.info("Start process sample data...")
        
        # 获取所有传感器的标定token - CARLA采集同步机制且无相对抖动 - 一个传感器一个标定
        sensor_calib_tokens = {
            sensor_name: 
                [c["token"] for c in self.nuscenes["calibrated_sensors"] if c["sensor_token"] == self.token_map[sensor_name]][0]
            for sensor_name in [
                "LIDAR_TOP",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_RIGHT",
                "CAM_BACK_LEFT"
            ]
        }
        
        # 按传感器类型分组存储样本数据（用于构建prev/next）
        sensor_data_map = defaultdict(list)

        # nuScenes中，一个sample对应的多个sample_data由于采集时间不完全对齐，对应不同的ego_pose
        for sample in tqdm(self.nuscenes["samples"], desc='Processing'):
            
            # 通过样本时间戳，最邻近的时间获取sample_data，实际上时间戳完全对齐
            sample_timestamp = sample["timestamp"]
            frame = self.timestamp_map[sample_timestamp]

            src_filename = f"{frame}.bin"
            dst_filename = f"{self.sequence:0>4}-{frame}-LIDAR_TOP-{sample_timestamp}.bin"
            # LiDAR数据
            lidar_data = {
                "token": generate_token(),
                "sample_token": sample["token"],
                "ego_pose_token": self.ego_pose_tokens[sample_timestamp],
                "calibrated_sensor_token": sensor_calib_tokens["LIDAR_TOP"],
                "filename": f"samples/LIDAR_TOP/{dst_filename}",
                # TODO: 使用Lidar的时间戳 - 文件名解析
                "timestamp": sample_timestamp,
                "fileformat": "pcd",
                "is_key_frame": True,
                "height": 0,
                "width": 0,
                "prev": "",
                "next": ""
            }

            self.nuscenes["sample_data"].append(lidar_data)
            sensor_data_map["LIDAR_TOP"].append(lidar_data)

            # process point cloud bin file, set ring index（5 demension）
            src_bin_path = os.path.join(self.carla_root, "sequences", self.sequence, "velodyne", src_filename)
            dst_bin_path = os.path.join(self.output_dir, "samples", "LIDAR_TOP", dst_filename)
            points = np.fromfile(src_bin_path, dtype=np.float32).reshape(-1, 4)
    
            # 添加第5列（ring index），全部设为0
            ring_indices = np.zeros((points.shape[0], 1), dtype=np.float32)
            points_with_ring = np.hstack((points, ring_indices))
            points_with_ring.tofile(dst_bin_path)

            # 6个相机数据
            for cam_name in [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_RIGHT",
                "CAM_BACK_LEFT"
            ]:
                
                src_filename = f"{frame}.jpg"
                dst_filename = f"{self.sequence:0>4}-{frame}-{cam_name}-{sample_timestamp}.jpg"

                cam_data = {
                    "token": generate_token(),
                    "sample_token": sample["token"],
                    "ego_pose_token": self.ego_pose_tokens[sample_timestamp],
                    "calibrated_sensor_token": sensor_calib_tokens[cam_name],
                    "filename": f"samples/{cam_name}/{dst_filename}",
                    # TODO: 使用camera的时间戳 - 文件名解析
                    "timestamp": sample_timestamp,
                    "fileformat": "jpg",
                    "is_key_frame": True,
                    "height": 900,
                    "width": 1600,
                    "prev": "",
                    "next": ""
                }

                self.nuscenes["sample_data"].append(cam_data)
                sensor_data_map[cam_name].append(cam_data)

                # 复制样本数据
                
                copy_file(
                    src=os.path.join(self.carla_root, "sequences", self.sequence, "image", cam_name, src_filename),
                    dst=os.path.join(self.output_dir, "samples", cam_name, dst_filename)
                )
        
        # 为每个传感器的数据构建prev/next链
        for sensor_name, data_list in sensor_data_map.items():
            # 按时间戳排序
            data_list.sort(key=lambda x: x["timestamp"])
            
            # 构建链表关系
            for i in range(len(data_list)):
                if i > 0:
                    data_list[i]["prev"] = data_list[i-1]["token"]
                if i < len(data_list) - 1:
                    data_list[i]["next"] = data_list[i+1]["token"]


    def _create_sample_annotation(self):
        """
            Create sample_annotation.json content from CARLA annotations
            {
                "token": "70aecbe9b64f4722ab3c230391a3beb8",
                "sample_token": "cd21dbfc3bd749c7b10a5c42562e0c42",
                "instance_token": "6dd2cbf4c24b4caeb625035869bca7b5",
                "visibility_token": "4",
                "attribute_tokens": [
                    "4d8821270b4a47e3a8a300cbec48188e"
                ],
                "translation": [
                    373.214,
                    1130.48,
                    1.25
                ],
                "size": [
                    0.621,
                    0.669,
                    1.642
                ],
                "rotation": [
                    0.9831098797903927,
                    0.0,
                    0.0,
                    -0.18301629506281616
                ],
                "prev": "a1721876c0944cdd92ebc3c75d55d693",
                "next": "1e8e35d365a441a18dd5503a0ee1c208",
                "num_lidar_pts": 5,
                "num_radar_pts": 0
            },
        
        """
        
        # Load the annotation file
        annotation_file = os.path.join(self.carla_root, "sequences", self.sequence, "labels.json")
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found at {annotation_file}")
            return
            
        with open(annotation_file) as f:
            all_frames_annotations = json.load(f)
        
        # Create a mapping from frame_id to annotations
        frame_annotations = {frame["frame_id"]: frame["objects"] for frame in all_frames_annotations}
        
        # Process annotations for each sample
        for sample in self.nuscenes["samples"]:
            # Convert timestamp to frame_id format (e.g., 123 -> "000123")
            timestamp = sample["timestamp"]
            frame_id = self.timestamp_map[timestamp]
            
            if frame_id not in frame_annotations:
                continue
                
            for obj in frame_annotations[frame_id]:
                # Convert rotation from CARLA to nuScenes format
                # CARLA rotation appears to be [pitch, yaw, roll] in radians
                # We'll use just the yaw component (index 1)

                # CARLA -> nuScenes coordinate
                roll_rad = obj["rotation"][0]
                pitch_rad = obj["rotation"][1]
                yaw_rad = obj["rotation"][2]

                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)
                q_carla = euler_to_quaternion(pitch_deg, -yaw_deg, roll_deg)
                q_nuscenes = carla_quaternion_to_nuscenes(q_carla)
                
                # Get category token
                category_token = next(
                    (c["token"] for c in self.nuscenes["category"] 
                    if c["name"] == obj["class"]),
                    None
                )
                if category_token is None:
                    print(f"Warning: Unknown category {obj['class']}, skipping object")
                    continue
                
                annotation_token = generate_token()

                # Create instance if not exists
                instance_token = self._get_or_create_instance(obj["object_id"], category_token, annotation_token)
                
                annotation_translation = [
                    obj["location"][0],
                    -obj["location"][1],
                    obj["location"][2]
                ]

                # Create annotation
                annotation = {
                    "token": annotation_token,
                    "sample_token": sample["token"],
                    "instance_token": instance_token,
                    "visibility_token": str(obj["occlusion"]),
                    "attribute_tokens": self._get_attribute_tokens(obj),
                    "translation": annotation_translation,
                    "size": [
                        obj["dimensions"][1],  # width (y)
                        obj["dimensions"][0],  # length (x)
                        obj["dimensions"][2]   # height (z)
                    ],
                    "rotation": q_nuscenes,
                    "prev": "",  # Will be filled later
                    "next": "",  # Will be filled later
                    "num_lidar_pts": 0,     # Placeholder - should be calculated from point cloud
                    "num_radar_pts": 0      # Placeholder - not used in CARLA
                }
                
                self.nuscenes["sample_annotation"].append(annotation)
        
        # Build prev/next links for each instance
        self._build_annotation_links()

    def _create_category_and_instance(self):
        """Create category.json content and initialize instances"""

        # Initialize instance tracking
        self.instance_map = {}  # Maps CARLA object_id to nuScenes instance_token

        # if Standard nuScenes categories had been create
        if os.path.exists(os.path.join(self.output_dir, "v1.0-trainval", "category.json")):
            with open(os.path.join(self.output_dir, "v1.0-trainval", "category.json"), "r") as f:
                categories = json.load(f)
                self.nuscenes["category"] = categories
                return

        # else create categories
        categories = [
            {"token": generate_token(), "name": "Pedestrian", "description": ""},
            {"token": generate_token(), "name": "Car", "description": ""},
            {"token": generate_token(), "name": "Bicycle", "description": ""},
            {"token": generate_token(), "name": "Motorcycle", "description": ""},
            {"token": generate_token(), "name": "Truck", "description": ""},
            {"token": generate_token(), "name": "TrafficLight", "description": ""},
            {"token": generate_token(), "name": "TrafficSigns", "description": ""},
            {"token": generate_token(), "name": "Static", "description": ""},
            {"token": generate_token(), "name": "Poles", "description": ""}
        ]
        self.nuscenes["category"] = categories
        
    def _get_or_create_instance(self, object_id, category_token, annotation_token):
        """Get or create an instance token for the given object_id"""
        if object_id not in self.instance_map:
            self.instance_map[object_id] = {
                "token": generate_token(),
                "category_token": category_token,
                "nbr_annotations": 0,
                "first_annotation_token": annotation_token,
                "last_annotation_token": annotation_token
            }
        else:
            self.instance_map[object_id]["nbr_annotations"] += 1
            self.instance_map[object_id]["last_annotation_token"] = annotation_token
        return self.instance_map[object_id]["token"]

    def _create_instance(self):
        self.nuscenes["instance"] = [instance for instance in self.instance_map.values()]

    def _get_attribute_tokens(self, obj):
        """Convert CARLA object properties to nuScenes attributes"""
        return self.nuscenes["attribute"][0]["token"]
    
    def _create_attribute(self):
        attribute = [
            {
                "token": generate_token(),
                "name": "cycle.with_rider",
                "description": "There is a rider on the bicycle or motorcycle."
            },
        ]
        self.nuscenes["attribute"] = attribute

    def _build_annotation_links(self):
        """Build prev/next links for annotations of each instance"""
        # Group annotations by instance
        instance_annotations = defaultdict(list)
        for ann in self.nuscenes["sample_annotation"]:
            instance_annotations[ann["instance_token"]].append(ann)
        
        # Sort annotations by timestamp and build links
        for instance_token, annotations in instance_annotations.items():
            annotations.sort(key=lambda x: next(
                s["timestamp"] for s in self.nuscenes["samples"] 
                if s["token"] == x["sample_token"]
            ))
            
            for i in range(len(annotations)):
                if i > 0:
                    annotations[i]["prev"] = annotations[i-1]["token"]
                if i < len(annotations) - 1:
                    annotations[i]["next"] = annotations[i+1]["token"]
         

    def _save_json_files(self):
        """保存所有JSON文件"""
        os.makedirs(os.path.join(self.output_dir, "v1.0-trainval"), exist_ok=True)
        
        # 保存核心文件
        core_files = {
            "log": "log",
            "sensor": "sensors",
            "calibrated_sensor": "calibrated_sensors",
            "ego_pose": "ego_poses",
            "scene": "scenes",
            "sample": "samples",
            "sample_data": "sample_data",
            "category": "category",
            "sample_annotation": "sample_annotation",
            "attribute": "attribute",
            "instance": "instance"
        }

        for filename, key in core_files.items():
            # 如果是sensors，calibrated_sensor，category，attribute，self.nuscenes[key]中保存的就是全量，直接写
            existing_data = []
            
            if key in ["sensors", "calibrated_sensors", "category", "attribute"]:
                existing_data = self.nuscenes[key]

            else:
                try:
                    with open(os.path.join(self.output_dir, "v1.0-trainval", f"{filename}.json"), "r") as f:
                        try:
                            existing_data = json.load(f)
                            existing_data.extend(self.nuscenes[key])
                        except json.JSONDecodeError:
                            existing_data = self.nuscenes[key]
                except FileNotFoundError:
                    existing_data = self.nuscenes[key]

            # 写入文件
            with open(os.path.join(self.output_dir, "v1.0-trainval", f"{filename}.json"), 'w') as f:
                json.dump(existing_data, f, indent=2)


if __name__ == "__main__":

    sequences = []

    converter = CarlaToNuScenesConverter(
        carla_root="./carla_data",
        nuscenes_output="./nuScenes",
        sequence="10"
    )
    converter.convert()
    print("Conversion completed!")