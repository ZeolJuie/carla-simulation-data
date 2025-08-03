import os
import sys
import csv
import time
import copy
import json
import math
import random
import argparse
from datetime import datetime
from queue import Queue
from queue import Empty

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import carla

import config
from sensors.camera import CameraSensor
from sensors.depth_camera import DepthCameraSensor
from sensors.semantic_camera import SemanticCameraSensor
from sensors.lidar import LidarSensor
from sensors.semantic_lidar import SemanticLidarSensor
from sensors.radar import RadarSensor
from sensors.imu import IMUSensor
from sensors.gnss import GNSSSensor
from utils.geometry_utils import *
from utils.folder_utils import *
from scenario.scenario_config import generate_scenario


def main(args):

    # parse args
    sequence_id = args.sequence
    scenario_id = args.scenario

    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    scenario = generate_scenario(scenario_id)
    
    world = client.get_world()

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)


        blueprint_library = world.get_blueprint_library()

        # 如果是自动控制，随机选择一个行人蓝图，并创建行人
        if not args.no_auto_controll:
            walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))

            # set start point and end point
            start_point = scenario.start_point
            end_point = scenario.end_points

            try:
                # generate walker
                walker = world.spawn_actor(walker_bp, carla.Transform(start_point, carla.Rotation(yaw=0)))
            except:
                print("RuntimeError: Spawn failed because of collision at spawn position")
                world.apply_settings(original_settings)
                exit()
        else:
            assert args.walker_id is not None, "Walker ID must be provided!"
            walker = world.get_actor(actor_id=int(args.walker_id))
            walker.set_location(scenario.start_point)
            walker_blueprint_id = walker.type_id
            walker_bp = blueprint_library.find(walker_blueprint_id)
        
        # 创建行人 AI 控制器
        if not args.no_auto_controll:
            controller_bp = blueprint_library.find('controller.ai.walker')
            walker_controller = world.spawn_actor(controller_bp, carla.Transform(), walker)

        world.tick()

        # 应用控制
        if not args.no_auto_controll:
            walker_controller.start()
            walker_controller.go_to_location(end_point)
            walker_controller.set_max_speed(0.8 + random.random())

        spec = world.get_spectator()

        # Main loop
        while True:
            # Tick the server
            world.tick()

            # 获取行人位置
            walker_location = walker.get_location()
            print(f"walker location: ({walker_location.x:.2f}, {walker_location.y:.2f}, {walker_location.z:.2f})")

            # 设置观察者的位置（行人后方 3 米，高度 2 米）
            spectator_location = walker_location + walker.get_transform().get_forward_vector() * -3.0 + carla.Location(z=2.0)  # 稍微抬高视角

            # 设置观察者的旋转（与行人同方向）
            spectator_rotation = walker.get_transform().rotation
            spectator_rotation.pitch = -20  # 稍微俯视（可选调整）

            # 更新 Spectator
            spec.set_transform(carla.Transform(spectator_location, spectator_rotation))


    finally:

        # 销毁 actor
        if not args.no_auto_controll:
            walker.destroy()
        
        # 还原 client world 设置
        world.apply_settings(original_settings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect Multimodel Data in CARLA World")
    parser.add_argument("sequence", type=str, help="Sequence_id")
    parser.add_argument("--scenario", type=str, help="Scenario_id")
    parser.add_argument("--no-auto-controll", action="store_true", default=False, help="Disable AI controller, default: False")
    parser.add_argument("--walker-id", type=str, help="If disable ai controller, must give a manual controll walker id")
    args = parser.parse_args()

    sequence_id = args.sequence


    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')