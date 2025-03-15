```
auto-driving-dataset-collection/
│
├── carla_simulation/
│   ├── __init__.py
│   ├── simulation_manager.py        # 管理仿真环境、场景切换等
│   ├── sensor_manager.py           # 管理各类传感器
│   ├── data_manager.py             # 处理传感器数据，存储
│   ├── walker_manager.py           # 控制行人（视障行人视角等）
│   ├── vehicle_manager.py          # 控制车辆（如有）
│   └── scenario_manager.py         # 定义并执行不同的仿真场景
│
├── config/
│   ├── config.py                   # 配置文件，定义传感器、场景、路径等参数
│   ├── sensors.json                # 各传感器配置（如分辨率、帧率、类型等）
│   ├── scenarios.json              # 不同场景配置文件
│
├── data/
│   ├── images/                     # 存储图像数据
│   ├── lidar/                      # 存储LiDAR数据
│   ├── depth/                      # 存储深度图
│   ├── gps_imu/                    # 存储GPS/IMU数据
│   └── logs/                       # 日志文件
│
├── scripts/
│   ├── run_simulation.py           # 运行仿真的脚本
│   ├── collect_data.py             # 采集数据的脚本
│   └── test_scenario.py            # 测试场景的脚本
│
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明文档


```
python ~/carla/PythonAPI/examples/generate_traffic.py --safe -n 200 -w 100 --asynch --port 2000
