from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化NuScenes对象
nusc = NuScenes(version='v1.0-trainval', dataroot='./nuScenes', verbose=True)

# 选择一个样本
sample = nusc.sample[2]  # 第10个样本

# 获取对应的LIDAR数据
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

# 加载点云
pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_data['token']))

# 可视化点云
plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter(pc.points[0, :], pc.points[1, :], pc.points[2, :], s=1, c=pc.points[2, :], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Point Cloud Visualization')
plt.show()