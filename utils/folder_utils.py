import os
from datetime import datetime


def create_folders(base_path, sequences_id):
    """
    根据序号创建文件夹，并在该文件夹下创建 image/ 和 velodyne/ 等子文件夹。

    参数:
        base_path (str): 基础路径，用于存放新创建的文件夹。
    """

    # 创建主文件夹路径
    main_folder = os.path.join(base_path, sequences_id)

    # 创建主文件夹
    os.makedirs(main_folder, exist_ok=True)
    print(f"create scene folder{main_folder}")

    # 创建子文件夹
    os.makedirs(os.path.join(main_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "depth"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "radar"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "calib"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "oxts"), exist_ok=True)
    

    print(f"create image folder{main_folder}/image")
    print(f"create image folder{main_folder}/depth")
    print(f"create image folder{main_folder}/semantic")
    print(f"create lidar folder：{main_folder}/velodyne")
    print(f"create lidar folder：{main_folder}/radar")
    print(f"create IMU/GPS folder{main_folder}/oxts")
    print(f"create calibreation folder{main_folder}/calib")
    
    return main_folder

