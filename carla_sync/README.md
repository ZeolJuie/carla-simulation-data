## Carla仿真
### Carla安装
### 1. Install Required System Dependency

Before downloading CARLA, install the necessary system dependency:

```
sudo apt-get -y install libomp5
```

### 2. Download the CARLA 0.9.15 Release

Download the CARLA_0.9.15.tar.gz file (approximately 16GB) from the official release:

```
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
```

### 3. Unpack CARLA to the Desired Directory

Unpack the downloaded file to `/opt/carla-simulator/`:

```bash
tar -xzvf CARLA_0.9.15.tar.gz -C /opt/carla-simulator/ # or your customized path
```

### 4. Install the CARLA Python Module

Finally, install the CARLA Python module and necessary dependencies:

`python -m pip install carla==0.9.15
python -m pip install -r /opt/carla-simulator/PythonAPI/examples/requirements.txt` 

if Shapely install failed, try to
```bash
sudo apt-get install libgeos-dev 
```



### 文件说明
* config/config.json: Unified management of sensor parameters (radar/camera, etc.) required
* occ_gen_so: Use the surroundocc-based method to generate occupancy gt, which can be used for carla-generated data or nuscenes-compliant data


### 代码运行
First, you need to ensure that the carla serve side is started:
```
cd {your carla dir}
./CarlaUE4.sh
```

Then in the carla_sync directory:
```bash
python dump_vehicle_bbox_6_camera_lidar.py
```