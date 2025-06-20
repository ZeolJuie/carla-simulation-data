# CARLA 服务器配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

FIXED_DELTA_SECONDS = 0.05

# 数据保存路径
DATA_DIR = "./carla_data/sequences"

# 行人控制参数
WALK_DISTANCE = 15.0  # 行人行走距离（米）
WALK_SPEED = 1.0  # 行人行走速度（米/秒）

# 摄像头参数
CAMERA_IMAGE_SIZE_X = 1600
CAMERA_IMAGE_SIZE_Y = 900
CAMERA_FOV = 120
CAMERA_SENSOR_TICK = 0.05

# 激光雷达参数
LIDAR_UPPER_FOV = 15.0
LIDAR_LOWER_FOV = -25.0
LIDAR_CHANNELS = 64.0
LIDAR_RANGE = 100.0
LIDAR_ROTATION_FREQUENCY = 20.0
LIDAR_POINTS_PER_SECOND = 1000000

SENSOR_TRANSFORM_X = 0.20
SENSOR_TRANSFORM_Y = 0
SENSOR_TRANSFORM_Z = 0.8
