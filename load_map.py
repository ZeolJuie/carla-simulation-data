import carla
import time

def main():
    # 1. 连接到 CARLA 服务端
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 为了防止 OOM 和卡顿，优先使用 Town10HD_Opt
    map_name = 'Town10HD_Opt'
    print(f"正在加载地图: {map_name} ...")
    world = client.load_world(map_name)
    
    # 稍微等待世界加载和静态网格体初始化
    time.sleep(2.0)

    print("正在查询静态车辆 ...")
    # 分别查询不同类型的车辆
    cars_bbs = world.get_level_bbs(carla.CityObjectLabel.Car)
    trucks_bbs = world.get_level_bbs(carla.CityObjectLabel.Truck)
    bus_bbs = world.get_level_bbs(carla.CityObjectLabel.Bus)
    motorcycles_bbs = world.get_level_bbs(carla.CityObjectLabel.Motorcycle)
    bicycles_bbs = world.get_level_bbs(carla.CityObjectLabel.Bicycle)
    
    # 将所有的 Bounding Box 合并到一个列表
    parked_vehicles_bbs = cars_bbs + trucks_bbs + bus_bbs + motorcycles_bbs + bicycles_bbs
    
    total_bbs = len(parked_vehicles_bbs)
    print(f"✅ 成功获取到 {total_bbs} 个静态车辆的 Bounding Box。")

    if total_bbs == 0:
        print("未找到任何车辆的 Bounding Box。这可能是因为该地图没有配置默认停放的车辆。")
        return

    # 3. 提取坐标并进行可视化
    debug = world.debug
    print("\n前 5 个静态车辆的坐标信息：")
    
    for i, bb in enumerate(parked_vehicles_bbs):
        # 打印前5个作为数据验证
        if i < 5:
            loc = bb.location
            ext = bb.extent
            print(f"  [{i}] 中心点: (X:{loc.x:.2f}, Y:{loc.y:.2f}, Z:{loc.z:.2f}) | 范围: (X:{ext.x:.2f}, Y:{ext.y:.2f}, Z:{ext.z:.2f})")
        
        # 绘制绿色边界框，厚度0.1，持续 30 秒
        debug.draw_box(
            box=bb, 
            rotation=bb.rotation, 
            thickness=0.1, 
            color=carla.Color(r=0, g=255, b=0), 
            life_time=30.0
        )

    print("\n👀 已在仿真器中绘制绿色 Bounding Box（持续30秒）。")
    print("您可以切换到 CARLA 服务端窗口，使用 WASD 和鼠标（或按 Alt+V）在场景中漫游查看标注是否贴合。")

if __name__ == '__main__':
    main()