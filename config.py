# CARLA 服务器配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

FIXED_DELTA_SECONDS = 0.05

# 数据保存路径
DATA_DIR = "data"

# 行人控制参数
WALK_DISTANCE = 15.0  # 行人行走距离（米）
WALK_SPEED = 1.0  # 行人行走速度（米/秒）

# 摄像头参数
CAMERA_IMAGE_SIZE_X = 800
CAMERA_IMAGE_SIZE_Y = 600
CAMERA_FOV = 120
CAMERA_SENSOR_TICK = 0.1

# 激光雷达参数
LIDAR_UPPER_FOV = 15.0
LIDAR_LOWER_FOV = -25.0
LIDAR_CHANNELS = 64.0
LIDAR_RANGE = 100.0
LIDAR_ROTATION_FREQUENCY = 20.0
LIDAR_POINTS_PER_SECOND = 500000