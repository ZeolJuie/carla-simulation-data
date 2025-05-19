# sensors/gnss.py

import os
import time

import csv
import numpy as np
import open3d as o3d
import carla

import config
from sensors.sensor import Sensor

class GNSSSensor(Sensor):
    def __init__(self, world, blueprint_library, walker, data_dir):
        super().__init__(world, blueprint_library, walker, data_dir, 'gnss')

    def _setup_sensor(self, blueprint_library, walker):

        gnss_bp = blueprint_library.find("sensor.other.gnss")
        gnss_bp.set_attribute("noise_alt_stddev", "0.2")        # 海拔噪声（可选）
        gnss_bp.set_attribute("noise_lat_stddev", "0.000001")   # 纬度噪声（可选）
        gnss_bp.set_attribute("noise_lon_stddev", "0.000001")   # 经度噪声（可选）

        gnss_transform = carla.Transform(carla.Location(x=config.SENSOR_TRANSFORM_X, z=config.SENSOR_TRANSFORM_Z))
        gnss = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=walker)

        # 创建 CSV 文件并写入表头
        file_path = f"{self.data_dir}/gnss_data.csv"
        self.csv_file = open(file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame", "timestamp",
            # GNSS 数据
            "gnss_latitude", "gnss_longitude", "gnss_altitude",
            # 车辆位置（Location）
            "location_x", "location_y", "location_z",
            # 车辆旋转（Rotation）
            "rotation_pitch", "rotation_roll", "rotation_yaw"
        ])

        return gnss_bp, gnss
    
    def _save_data(self, sensor_data):
        """
        保存 gnss 为csv格式 并写入磁盘。
        """
        
        transform = sensor_data.transform
        location = transform.location
        rotation = transform.rotation

        # 写入 CSV
        self.csv_writer.writerow([
            # 时间戳
            sensor_data.frame, sensor_data.timestamp,
            # 经纬度 高度
            sensor_data.latitude, sensor_data.longitude, sensor_data.altitude,
            # 行人位置（Location）
            location.x, location.y, location.z,
            # 行人旋转（Rotation）
            rotation.pitch, rotation.roll, rotation.yaw
        ])
        print(f"Saved: GNSS=({sensor_data.latitude}, {sensor_data.longitude}, {sensor_data.altitude}), "
            f"Location=({location.x}, {location.y}, {location.z}), "
            f"Rotation=({rotation.pitch}, {rotation.roll}, {rotation.yaw})")
        
    def destroy(self):
        super().destroy()

        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
        
