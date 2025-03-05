import carla

import time
import os
import random
import queue

import config
from sensors.camera import CameraSensor
from sensors.lidar import LidarSensor
from sensors.imu import IMUSensor
from actors.walker import Walker

def main():
    # 连接到 CARLA 服务器
    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT)
    world = client.get_world()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 创建传感器数据队列
    sensor_queue = queue.Queue()

    # 获取蓝图库
    blueprint_library = world.get_blueprint_library()

    # 生成行人
    spawn_points = world.get_map().get_spawn_points()
    # 随机生成初始位置
    walker = Walker(world, blueprint_library, random.choice(spawn_points))

    
    sensor_list = []

    # 设置传感器
    camera = CameraSensor(world, blueprint_library, walker.walker, config.DATA_DIR)
    sensor_list.append(camera)
    lidar = LidarSensor(world, blueprint_library, walker.walker, config.DATA_DIR)
    sensor_list.append(lidar)
    imu = IMUSensor(world, blueprint_library, walker.walker, config.DATA_DIR)
    sensor_list.append(imu)

    

    # 启动传感器
    # camera.start(sensor_queue, "camera")
    # lidar.start(sensor_queue, "lidar")
    # imu.start(sensor_queue, "imu")

    # 设置行人目标点
    initial_location = walker.get_location()
    target_location = carla.Location(
        x=initial_location.x + config.WALK_DISTANCE,
        y=initial_location.y,
        z=initial_location.z
    )

    walker.set_target_location(target_location)

    spec = world.get_spectator()


    try:
        while True:

            # 推进仿真时间步
            world.tick()

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # 获取行人位置
            walker_location = walker.get_location()
            print(f"行人位置: ({walker_location.x:.2f}, {walker_location.y:.2f}, {walker_location.z:.2f})")

            # 更新BEV视角
            spec.set_transform(carla.Transform(walker.get_transform().location + carla.Location(z=10), carla.Rotation(pitch=-90)))

            time.sleep(0.1)




    except KeyboardInterrupt:
        print("程序退出")

    finally:
        # 销毁传感器
        for sensor in sensor_list:
            sensor.destroy()

        # 销毁行人
        walker.destroy()

        # 将world重新设置会异步模式
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

if __name__ == "__main__":
    main()