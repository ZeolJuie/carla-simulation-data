# 测试生成大量行人是否会发生碰撞
"""
    测试生成大量行人是否会发生碰撞 - 不会碰撞
    再单独生成一个行人 并切换到该行人视角 测试其是否会碰撞 - 不会碰撞
    通过generate_traffic.py生成行人 再在其他脚本中单独生成一个行人 并切换到该行人视角 测试其是否会碰撞 - 会碰撞
        - 是同步异步的问题吗 - 理论上不是
        - 有没有其他解决方式 
            - generate_traffic.py不生成行人 通过采集脚本生成行人 需要验证行人和车是否会碰撞 - 会避让行人 但也有交通事故发生 需要生成数量适当 - 可行
        - github上提出issue
"""

import carla
import random


def generate_pedestrians(world, pedestrians_num):

    blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    
    # 获取所有可能的生成点
    spawn_points = []
    for i in range(pedestrians_num):
        loc = world.get_random_location_from_navigation()
        print(loc)
        if loc is not None:
            spawn_point = carla.Transform(loc)
            spawn_points.append(spawn_point)

    # 生成300个行人
    pedestrians = []
    controllers = []
    
    for i in range(pedestrians_num):
        try:
            # 随机选择行人蓝图
            blueprint = random.choice(blueprints)
            
            # 生成行人
            pedestrian = world.spawn_actor(blueprint, spawn_points[i])
            pedestrians.append(pedestrian)
            
            # 为行人创建AI控制器
            controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
            controllers.append(controller)
            
            # 启动控制器，设置行人随机行走
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            
        except Exception as e:
            print(f"Error spawning pedestrian {i}: {e}")
    
    print(f"Successfully spawned {len(pedestrians)} pedestrians")

    return pedestrians, controllers


def main():
    # 连接到CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 获取世界对象
    world = client.get_world()

    # 获取行人蓝图
    blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')

    pedestrians_num = 100
    
    pedestrians, controllers = generate_pedestrians(world, pedestrians_num)

    # 创建观察用的行人
    try:
        # 选择观察行人蓝图
        obs_blueprint = random.choice(blueprints)
        
        # 获取新的生成点
        obs_loc = world.get_random_location_from_navigation()

        spawn_points = [world.get_random_location_from_navigation() for _ in range(300)]

        # sort x[-125, 120]   y[-80, 150]
        spawn_points = sorted(spawn_points, key=lambda point: (-point.x, -point.y))

        # set start point and end point
        start_point = carla.Location(x=117.118011, y=-8.054474, z=0.158620)
        end_point = carla.Location(x=80.076370, y=37.853725, z=0.158620)

        obs_spawn_point = carla.Transform(start_point)
        
        # 生成观察行人
        obs_pedestrian = world.spawn_actor(obs_blueprint, obs_spawn_point)
        
        pedestrians.append(obs_pedestrian)
        
        # 为观察行人创建AI控制器
        obs_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        obs_controller = world.spawn_actor(obs_controller_bp, carla.Transform(), obs_pedestrian)
        controllers.append(obs_controller)
        
        # 启动控制器，设置行人随机行走
        obs_controller.start()
        obs_controller.go_to_location(end_point)

        world.tick()
        
        # 创建观察视角
        spectator = world.get_spectator()
        
        # 保持脚本运行，让行人继续行走
        try:
            while True:
                world.tick()
                spectator.set_transform(carla.Transform(obs_pedestrian.get_transform().location + carla.Location(z=5), carla.Rotation(pitch=-90)))
        except KeyboardInterrupt:
            print("Destroying pedestrians and controllers...")
            for controller in controllers:
                controller.stop()
            client.apply_batch([carla.command.DestroyActor(x) for x in pedestrians])
            client.apply_batch([carla.command.DestroyActor(x) for x in controllers])
            
    except Exception as e:
        print(f"Error creating observer pedestrian: {e}")
        for controller in controllers:
            controller.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in pedestrians])
        client.apply_batch([carla.command.DestroyActor(x) for x in controllers])

if __name__ == '__main__':
    main()