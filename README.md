

数据处理

- 遮挡程度 
  
```
python data_process.py
```
- Static细分类

    demesions

数据集结构
```
nuScenes/
├── maps/
├── samples/          # 关键帧传感器数据
├── sweeps/           # 非关键帧传感器数据
├── v1.0-trainval/    # 元数据和标注
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   ├── visibility.json
```


# Data Preparation
## Data Collection
- Start CARLA Server
    ```
    ./CarlaUE4.sh -quality-level=Epic
    ./CarlaUE4.sh -RenderOffScreen
    ```
- Load Map
    ```
    python ~/carla/CARLA_0.9.15/PythonAPI/util/config.py -m Town01
    python ~/carla/PythonAPI/util/config.py -m Town01
    ```
- Generate Traffic
    ```
    python scenario/generate_traffic.py -n 100 -w 0 --asynch --port 2000 --filterv car 75 --filterv bicycle 20 --filterv truck 5
    ```
- Start Collect
  - 自动采集：行人可以从路线的起始点走到终点并完成避障，手动控制下无法走上台阶
    ```
    python start_sampling.py 01 --scenario 01
    ```
  - 手动采集
    ```
    # 开启手动采集视窗
    python manual_control.py
    # 记录手动采集的行人ID（Walker ID）
    python start_sampling.py <sequence_id> --scenario <scenario_id> --no-auto-controll --walker-id <walker_id>
    ```

## Data Process
- Format Convert to nuScenes Style
    
    ```
    # copy data to nuScenes dataset root
    scp -r ./carla_data/sequences/10/image/* ./nuScenes/samples/
    scp -r ./carla_data/sequences/10/velodyne/ ./nuScenes/samples/LIDAR_TOP
    
    python utils/converter/carla_to_nuscenes_converter.py 
    ```


NOTICE
- ego的rotation信息是角度制，labels中的rotation是弧度制