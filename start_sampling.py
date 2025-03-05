import glob
import os
import sys
import random
import time
import copy
from queue import Queue
from queue import Empty

import numpy as np
import open3d as o3d

import carla

from datetime import datetime


def create_folders(base_path):
    """
    根据当前时间创建文件夹，并在该文件夹下创建 image/ 和 lidar/ 子文件夹。
    
    参数:
        base_path (str): 基础路径，用于存放新创建的文件夹。
    """
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间字符串，例如：2025-03-05_14-30-00
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 创建主文件夹路径
    main_folder = os.path.join(base_path, time_str)

    # 创建主文件夹
    os.makedirs(main_folder, exist_ok=True)
    print(f"create scene folder{main_folder}")

    # 创建子文件夹
    os.makedirs(os.path.join(main_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "lidar"), exist_ok=True)

    print(f"create image folder{main_folder}/image")
    print(f"create lidar folder：{main_folder}/lidar")

    return main_folder

main_folder = create_folders('./data')


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    try:
        if sensor_name == "camera":
            # 处理 RGB 图像
            save_rgb_image(sensor_data)
        elif sensor_name == "lidar":
            # 处理 LiDAR 点云
            save_lidar_point_cloud(sensor_data)
        elif sensor_name == "imu":
            # 处理 IMU 数据
            save_imu_data(sensor_data)
        
        # 将数据帧和传感器名称放入队列
        sensor_queue.put((sensor_data.frame, sensor_name))
    except Exception as e:
        print(f"Error processing {sensor_name} data: {e}")


def save_rgb_image(sensor_data):
    """
    保存 RGB 图像到磁盘。
    """
    file_path = os.path.join(f"{main_folder}/image", '%06d.png' % sensor_data.frame)
    sensor_data.save_to_disk(file_path)
    print(f"Saved RGB image to {file_path}")


def save_lidar_point_cloud(sensor_data):
    """
    保存 LiDAR 点云到磁盘。
    """
    data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype("f4")))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    points = data[:, :-1]
    # TODO: 这里为什么取负号？
    points[:, 1] = -points[:, 1]

    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)

    file_path = os.path.join(f"{main_folder}/lidar", '%06d.ply' % sensor_data.frame)
    o3d.io.write_point_cloud(file_path, o3d_point_cloud)

    print(f"Saved LiDAR point cloud to {file_path}")


def save_imu_data(sensor_data):
    """
    保存 IMU 数据到磁盘。
    """
    print(f"Saved IMU data")


def main():

    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    

    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)


        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()

        blueprint_library = world.get_blueprint_library()

        # 随机选择一个行人蓝图
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))

        # 获取地图的推荐点
        spawn_points = world.get_map().get_spawn_points()

        # 随机选择一个生成点
        # spawn_point = random.choice(world.get_map().get_spawn_points())
        spawn_point = world.get_random_location_from_navigation()

        # 生成行人
        # walker = world.spawn_actor(walker_bp, random.choice(spawn_points))
        # 获取所有行人
        walker = random.choice(world.get_actors().filter('walker.*'))

        # create all the sensors and keep them in a list for convenience.
        sensor_list = []

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        imu_bp = blueprint_library.find('sensor.other.imu')


        # 设置摄像头
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '120')
        camera_bp.set_attribute('sensor_tick', '0.1')  # 设置摄像头频率为 10 Hz
        camera_transform = carla.Transform(carla.Location(x=0.0, z=1.5))  # 摄像头位置
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=walker)

        camera.listen(lambda data: sensor_callback(data, sensor_queue, "camera"))
        sensor_list.append(camera)


        # 设置激光雷达
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

        lidar_bp.set_attribute("upper_fov", str(15.0))
        lidar_bp.set_attribute("lower_fov", str(-25.0))
        lidar_bp.set_attribute("channels", str(64.0))
        lidar_bp.set_attribute("range", str(100.0))
        lidar_bp.set_attribute("rotation_frequency", str(20.0 / 1))
        lidar_bp.set_attribute("points_per_second", str(500000))

        # 激光雷达位置
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=walker)

        lidar.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))
        sensor_list.append(lidar)

        # imu同雷达绑定
        imu_transform = lidar_transform
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=walker)
        
        imu.listen(lambda data: sensor_callback(data, sensor_queue, "imu"))
        sensor_list.append(imu)

        # 创建行人 AI 控制器
        controller_bp = blueprint_library.find('controller.ai.walker')
        walker_controller = world.spawn_actor(controller_bp, carla.Transform(), walker)

        # important! wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        walker_controller.start()

        # 应用控制
        walker_controller.go_to_location(world.get_random_location_from_navigation())

        walker_controller.set_max_speed(1 + random.random())

        spec = world.get_spectator()

        # Main loop
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # 获取行人位置
            walker_location = walker.get_location()
            print(f"walker location: ({walker_location.x:.2f}, {walker_location.y:.2f}, {walker_location.z:.2f})")
            
            spec.set_transform(carla.Transform(walker.get_transform().location + carla.Location(z=10), carla.Rotation(pitch=-90)))

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("    Some of the sensor information is missed")


    finally:
        world.apply_settings(original_settings)

        walker.destroy()

        for sensor in sensor_list:
            sensor.destroy()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')