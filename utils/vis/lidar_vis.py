import open3d as o3d
import numpy as np


def get_label_color(label):
    """
    根据CARLA语义标签返回对应的RGB颜色
    参数:
        label (int): CARLA语义标签值
    返回:
        tuple: (R, G, B) 颜色值，范围0-255
    """
    color_map = {
        0: (0, 0, 0),        # Unlabeled
        1: (128, 64, 128),   # Roads
        2: (244, 35, 232),   # SideWalks
        3: (70, 70, 70),     # Building
        4: (102, 102, 156),  # Wall
        5: (190, 153, 153),  # Fence
        6: (153, 153, 153), # Pole
        7: (250, 170, 30),  # TrafficLight
        8: (220, 220, 0),   # TrafficSign
        9: (107, 142, 35),  # Vegetation
        10: (152, 251, 152),# Terrain
        11: (70, 130, 180), # Sky
        12: (220, 20, 60),  # Pedestrian
        13: (255, 0, 0),    # Rider
        14: (0, 0, 142),    # Car
        15: (0, 0, 70),     # Truck
        16: (0, 60, 100),   # Bus
        17: (0, 80, 100),   # Train
        18: (0, 0, 230),    # Motorcycle
        19: (119, 11, 32),  # Bicycle
        20: (110, 190, 160),# Static
        21: (170, 120, 50), # Dynamic
        22: (55, 90, 80),   # Other
        23: (45, 60, 150),  # Water
        24: (157, 234, 50), # RoadLine
        25: (81, 0, 81),    # Ground
        26: (150, 100, 100),# Bridge
        27: (230, 150, 140),# RailTrack
        28: (180, 165, 180) # GuardRail
    }
    # 默认返回白色(用于未知标签)
    return color_map.get(label, (255, 255, 255)) 

def visualize_ply_file(ply_file):
    """可视化PLY格式的点云文件"""
    try:
        pcd = o3d.io.read_point_cloud(ply_file)
        if not pcd.points:
            raise ValueError("PLY文件没有包含点云数据")
            
        o3d.visualization.draw_geometries([pcd],
                                        window_name="PLY Point Cloud",
                                        width=1024,
                                        height=768,
                                        left=50,
                                        top=50)
    except Exception as e:
        print(f"无法加载PLY文件: {e}")

def visualize_with_open3d(bin_file):
    # 加载数据
    
    data = np.fromfile(bin_file, dtype=np.float32)
    points = data.reshape(-1, 4)[:, :3]
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置点坐标（y已经取反）
    pcd.points = o3d.utility.Vector3dVector(points)
        
    # 可视化
    o3d.visualization.draw_geometries([pcd],
                                    window_name="LiDAR Point Cloud (Y Inverted)",
                                    width=1024,
                                    height=768,
                                    left=50,
                                    top=50)

# 调用示例
visualize_with_open3d("carla_data/sequences/10/velodyne/395450.bin")
# visualize_ply_file("carla_data/sequences/04/velodyne/043357.ply")
