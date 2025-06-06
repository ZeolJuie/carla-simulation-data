# sensors/imu.py

import os
import time

import csv
import numpy as np
import carla

import config
from sensors.sensor import Sensor


class IMUSensor(Sensor):
    def __init__(self, world, blueprint_library, walker, data_dir):
        super().__init__(world, blueprint_library, walker, data_dir, 'imu')

    def _setup_sensor(self, blueprint_library, walker):
       
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute("noise_accel_stddev_x", "0.01")  # X 轴加速度噪声
        imu_bp.set_attribute("noise_accel_stddev_y", "0.01")  # Y 轴加速度噪声
        imu_bp.set_attribute("noise_accel_stddev_z", "0.01")  # Z 轴加速度噪声
        imu_bp.set_attribute("noise_gyro_stddev_x", "0.001")  # X 轴角速度噪声
        imu_bp.set_attribute("noise_gyro_stddev_y", "0.001")  # Y 轴角速度噪声
        imu_bp.set_attribute("noise_gyro_stddev_z", "0.001")  # Z 轴角速度噪声

        # imu同行人绑定
        imu_transform = carla.Transform(carla.Location(x=config.SENSOR_TRANSFORM_X, z=config.SENSOR_TRANSFORM_Z))
        imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=walker)

        # 创建 CSV 文件并写入表头
        file_path = f"{self.data_dir}/imu_data.csv"
        self.csv_file = open(file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame",
            # 加速度（m/s²）
            "accel_x", "accel_y", "accel_z",
            # 角速度（rad/s）
            "gyro_x", "gyro_y", "gyro_z",
            # 方向（弧度）
            "compass"
        ])

        return imu_bp, imu
    
    def _save_data(self, sensor_data):
        """
        保存 IMU 数据到磁盘。
        """

        self.csv_writer.writerow([
            sensor_data.frame,
            # 加速度（m/s²）
            sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z,
            # 角速度（rad/s）
            sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z,
            # 罗盘方向（弧度）
            sensor_data.compass
        ])
        # print(f"IMU Data: Accel=({sensor_data.accelerometer.x:.3f}, {sensor_data.accelerometer.y:.3f}, {sensor_data.accelerometer.z:.3f}), "
        #     f"Gyro=({sensor_data.gyroscope.x:.3f}, {sensor_data.gyroscope.y:.3f}, {sensor_data.gyroscope.z:.3f}), "
        #     f"Compass={sensor_data.compass:.3f}")

        print(f"Saved IMU data, frame: {sensor_data.frame}")

    def destroy(self):
        super().destroy()

        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
    