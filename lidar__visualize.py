import open3d as o3d
import os
import time

def visualize_ply_files(folder_path, interval=0.1):
    """
    依次可视化文件夹中的 .ply 文件
    :param folder_path: 包含 .ply 文件的文件夹路径
    :param interval: 每帧点云的显示间隔时间（秒）
    """
    # 获取文件夹中的所有 .ply 文件
    ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    ply_files.sort()  # 按文件名排序

    if not ply_files:
        print(f"文件夹 {folder_path} 中没有 .ply 文件。")
        return

    # 创建 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云连续可视化")

    # 初始化点云对象
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)

    # 依次加载并显示每个 .ply 文件
    for ply_file in ply_files:
        # 加载点云数据
        file_path = os.path.join(folder_path, ply_file)
        new_point_cloud = o3d.io.read_point_cloud(file_path)


        if not new_point_cloud.has_points():
            print(f"文件 {ply_file} 中没有点云数据。")
            continue

        # 更新点云对象
        point_cloud.points = new_point_cloud.points
        if new_point_cloud.has_colors():
            point_cloud.colors = new_point_cloud.colors

        # 更新可视化窗口
        vis.add_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # 打印当前文件名
        print(f"显示点云文件: {ply_file}")

        # 等待指定时间
        time.sleep(interval)

    # 关闭可视化窗口
    vis.destroy_window()

if __name__ == "__main__":
    # 点云文件夹路径
    ply_folder = "data"  # 替换为你的 .ply 文件所在文件夹路径

    # 可视化点云文件
    visualize_ply_files(ply_folder, interval=0.2)  # 每帧间隔 0.1 秒