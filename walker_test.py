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

import cv2

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
    os.makedirs(os.path.join(main_folder, "image_label"), exist_ok=True)

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
        sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))
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


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
        构建一个 相机内参矩阵(Intrinsic Matrix), 用于将3D空间中的点投影到2D图像平面上
        input:
            w: width of image
            h: height of image
            fov: Field of View eg:120/90
            is_behind_camera: 于指示是否处理相机后方的点, 如果为True, 则焦距f取负值

        output:
            K = [   fx  0   cx
                    0   fy  cy
                    0   0   1   ]
            
            generally: fx = fy = f
            f = w / (2.0 * np.tan(fov * np.pi / 360.0))

            cx, cy: 图像的主点, 通常是图像中心

    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    """
        Calculate 2D projection of 3D coordinate
        :loc - a carla.Position object
        :K - projection matrix  (3D point -> 2D point)
            将相机坐标系下的3D点投影到2D图像平面 (内参)
        :w2c - world to camera matrix
            描述相机的位置和姿态    (外参)
            将世界坐标系下的3D点 转换到 相机坐标系下的3D点
            [   R(3x3)  T(3X1)
                0(1x3)  1     ]
    """

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False


def get_image_point(loc, K, w2c):
    # 计算三维坐标的二维投影

    # 格式化输入坐标（loc 是一个 carla.Position 对象）
    point = np.array([loc.x, loc.y, loc.z, 1])

    # 转换到相机坐标系
    point_camera = np.dot(w2c, point)

    # 将坐标系从 UE4 的坐标系转换为标准坐标系（y, -z, x），同时移除第四个分量
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # 使用相机矩阵进行三维到二维投影
    point_img = np.dot(K, point_camera)

    # 归一化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


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

        # traffic_manager = client.get_trafficmanager()
        # traffic_manager.set_synchronous_mode(True)


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
        walker = world.spawn_actor(walker_bp, random.choice(spawn_points))
        # 获取所有行人
        # walker = random.choice(world.get_actors().filter('walker.*'))
        
        pedestrians = []
        pedestrians_num = 0
        for i in range(pedestrians_num):
            pedestrians_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
            try:
                pedestrian = world.spawn_actor(walker_bp, random.choice(spawn_points))
                if pedestrian:
                    pedestrians.append(pedestrian)
            except Exception:
                print(Exception)
        
        pedestrian_controllers = []
        for p in pedestrians:
            pedestrian_bp = blueprint_library.find('controller.ai.walker')
            pedestrian_controller = world.spawn_actor(pedestrian_bp, carla.Transform(), p)
            pedestrian_controllers.append(pedestrian_controller)


        # create all the sensors and keep them in a list for convenience.
        sensor_list = []

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        imu_bp = blueprint_library.find('sensor.other.imu')


        # 设置摄像头
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '120')
        # camera_bp.set_attribute('sensor_tick', '0.1')  # 设置摄像头频率为 10 Hz
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


        for p_c in pedestrian_controllers:
            p_c.start()
            # 应用控制
            p_c.go_to_location(world.get_random_location_from_navigation())
            p_c.set_max_speed(1 + random.random())


        spec = world.get_spectator()

        # Remember the edge pairs
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

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


                # Reshape the raw data into an RGB array
                image = s_frame[2]
                img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

                # Get the camera matrix 
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

                world_2_lidar =  np.array(lidar.get_transform().get_inverse_matrix())


                npcs = [p for p in world.get_actors().filter('*pedestrian*')] + [v for v in world.get_actors().filter('*vehicle*')]

                for npc in npcs:
                    # 获取3D Bounding Box
                    bbox = npc.bounding_box

                    transform = npc.get_transform()

                    # 计算 walker 和 npc 之间的距离
                    dist = transform.location.distance(walker.get_transform().location)

                    # 筛选距离在50米以内的车辆
                    if dist < 50:

                        
                        # 获取 Bounding Box 的中心点（世界坐标系）
                        bbox_center = transform.transform(bbox.location)
                        global_bbox_center = np.array([bbox_center.x, bbox_center.y, bbox_center.z, 1])
                        # world coordinate -> lidar coordinate
                        lidar_bbox_center = np.dot(w2p, global_bbox_centeroint)

                        # 获取 Bounding Box 的尺寸
                        bbox_extent = bbox.extent

                        # 存储 Bounding Box 数据
                        bbox_data[] = {
                            'object_id': npc.id,
                            'frame_id': w_frame
                            'bbox_center': lidar_bbox_center[0: 2],
                            'bbox_extent': [bbox_extent.x, bbox_extent.y, bbox_extent.z],
                            # 'rotation': [rotation.pitch, rotation.yaw, rotation.roll]
                        }


                        # 计算行人前进方向与npc之间的向量的点积，
                        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA

                        # 行人前进方向
                        forward_vec = walker.get_transform().get_forward_vector()
                        # 行人和npc之间的向量
                        ray = npc.get_transform().location - walker.get_transform().location

                        # 点积>0说明两个向量夹角<90
                        # 该npc在行人的前进方向的视野里
                        if forward_vec.dot(ray) > 0:

                            # get the 3D coordinates of the bounding box in world coordinates
                            verts = [v for v in bbox.get_world_vertices(npc.get_transform())]

                            # calculate 2D point in camera image canvas
                            for edge in edges:
                                
                                p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                                p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                                p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                                if not p1_in_canvas and not p2_in_canvas:
                                    continue

                                ray0 = verts[edge[0]] - camera.get_transform().location
                                ray1 = verts[edge[1]] - camera.get_transform().location
                                cam_forward_vec = camera.get_transform().get_forward_vector()

                                # One of the vertex is behind the camera
                                if not (cam_forward_vec.dot(ray0) > 0):
                                    p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                                if not (cam_forward_vec.dot(ray1) > 0):
                                    p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                                cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)
                        

                # cv2.imshow('ImageWindowName', img)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                        
                        # NOTICE: 这里写入的是T-1帧的图片
                        file_path = os.path.join(f"{main_folder}/image_label", '%06d.png' % s_frame[0])
                        cv2.imwrite(file_path, img)

            except Empty:
                print("    Some of the sensor information is missed")


    finally:
        world.apply_settings(original_settings)

        walker.destroy()

        for pedestrian in pedestrians:
            pedestrian.destroy()
    

        for sensor in sensor_list:
            sensor.destroy()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')