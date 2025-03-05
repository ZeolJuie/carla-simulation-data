import os
import open3d as o3d
import numpy as np

def load_ply_files(folder_path):
    """
    加载指定文件夹中的所有PLY文件
    """
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.ply')])
    tf_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
    point_clouds = []
    for p_file, tf_file in zip(files, tf_files):
        print(f"Loading points:{p_file}, tf: {tf_file}")
        # lidar
        point_cloud = o3d.io.read_point_cloud(p_file)
        points = np.asarray(point_cloud.points)

        # transform_matrix
        transform_matrix = np.load(tf_file)

        # lidar -> world
        rotation_matrix = transform_matrix[:3, :3]  # 3x3 旋转矩阵
        translation_vector = transform_matrix[:3, 3]  # 平移向量
        points_world = np.dot(rotation_matrix, points.T).T + translation_vector

        # 更新点云对象
        point_cloud.points = o3d.utility.Vector3dVector(points_world)
        point_clouds.append(point_cloud)
    return point_clouds



def visualize_point_clouds(point_clouds, fps=10):
    """
    自动播放连续帧点云
    :param fps: 每秒播放的帧数
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer", width=800, height=600)

    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1

    # 初始化可视化对象
    current_index = 0
    point_cloud = point_clouds[current_index]
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # 设置点云颜色
    vis.add_geometry(point_cloud)

    def update_geometry(vis):
        nonlocal current_index
        current_index = (current_index + 1) % len(point_clouds)
        point_cloud.points = point_clouds[current_index].points
        point_cloud.colors = point_clouds[current_index].colors
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

    # 设置定时器自动更新
    vis.register_animation_callback(update_geometry)

    # 设置播放速度（通过帧率控制）
    vis.run()  # 这里会自动开始播放
    vis.destroy_window()

if __name__ == "__main__":
    folder_path = "data"  # 替换为你的PLY文件夹路径
    point_clouds = load_ply_files(folder_path)
    if point_clouds:
        visualize_point_clouds(point_clouds, fps=10)  # 设置每秒播放10帧
    else:
        print("No PLY files found in the specified folder.")