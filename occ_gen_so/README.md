## Occupancy ground truth generation

### Install Chamfer Distance.
```shell
cd ./occ_gen_so/chamfer_dist
python setup.py install
```
 
### Data
Here we provide a data example for generating occ：../carla_sync/template_record \
The format is as follows：
```
template_record/
├── point_cloud/ (激光点云语义分割数据，该数据Nx4, 其中4是xyzc，c是语义类别)
│   ├── {time_stamp_0}.npy
│   ├── {time_stamp_1}.npy
│   ├── ...
├── 3d_bbox/
│   ├── bbox_{time_stamp_0}.npy (bounding box of the object, Nx7, [x, y, z, x_size, y_size, z_size, rz], (x, y, z) is the bottom center)
│   ├── bbox_{time_stamp_1}.npy
│   ├── ...
│   ├── object_category_{time_stamp_0}.npy (semantic category of the object, N, ps: 目前和点云的类别index未统一)
│   ├── object_category_{time_stamp_1}.npy
│   ├── ...
│   ├── boxes_token_{time_stamp_0}.npy (Unique bbox codes used to combine the same object in different frames, N)
│   ├── boxes_token_{time_stamp_1}.npy
│   ├── ...
├── transform_infor/ 雷达相对于世界坐标系的外参，4x4的矩阵，用于多帧对齐
│   ├── lidar_2_world_{time_stamp_0}.npy
│   ├── lidar_2_world_{time_stamp_1}.npy
│   ├── ...
├── pkl/ 用于对齐nuscenes格式的信息，其中内容比较多，具体如下：
│   ├── {time_stamp_0}.pkl
│   ├── {time_stamp_1}.pkl
│   ├── ...
```
The content and format of the .pkl file：
```
    a. 'infos': 是只有一个元素的list
        i. 'lidar_path': 点云文件路径，.bin类型，原size为Nx5，x, y, z, intensity, ring, index
        ii. 'occ_path': occ文件路径
        iii. 'token': 唯一标识token
        iv. 'sweeps': ???
        v. 'lidar2ego_translation': 雷达到ego车的位移外参, x y z
        vi. 'lidar2ego_rotation': 雷达到ego车的旋转外参, 四元数
        vii. 'ego2global_translation': ego车到全局坐标系的位移外参, x y z
        viii. 'ego2global_rotation': ego车到全局坐标系的旋转外参, 四元数
        ix. 'timestamp': 时间戳，以雷达点云生成时间点为准
        x. 'gt_boxes': Mx7的3d框信息，x, y, z, l, w, h, rz (z轴旋转弧度)
        xi. 'gt_names': 长度为M的array，为3d框中object的类别信息，np.str_
        xii. 'gt_velocity': Mx2的object速度信息，分为x, y两个方向
        xiii. 'num_lidar_pts': 长度为M的array，代表各3d框内包含的lidar点云数量
        xiv. 'num_radar_pts': 长度为M的array，代表各3d框内包含的radar点数量？ 
        xv. 'valid_flag': 长度为M的array，内为bool值，代表各3d框是否有效(lidar点云数不为0)
        xvi. 'ann_infos': ???
        xvii. 'scene_token': 场景唯一标识符
        xviii. 'scene_name'：场景名称
        xix. 'cams': 字典，包含相机相关信息，key值包括：["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]，对于每个相机，其value又是一个字典，具体如下
            1. 'data_path',  对应相机的图像路径
            2. 'type',  对应相机名称(如'CAM_FRONT')
            3. 'sample_data_token', 唯一标识符
            4. 'sensor2ego_translation', 相机到ego车辆的外参位移
            5. 'sensor2ego_rotation', 相机到ego车辆的外参旋转
            6. 'ego2global_translation', ego车辆到世界坐标系的外参位移，和外面lidar的可能不一样，因为相机和lidar数据采样的时刻不同
            7. 'ego2global_rotation', ego车辆到世界坐标系的外参旋转
            8. 'timestamp', 图像获取时的时间戳
            9. 'sensor2lidar_rotation', 相机到lidar的外参旋转
            10. 'sensor2lidar_translation', 相机到lidar的外参位移
            11. 'cam_intrinsic', 相机内参
    b. 'metadata': {'version': xxxx} 主要用于存储数据集版本等
```
### Generation
Under the dir: "./occ_gen_so"
``` 
python gen_occ_chunk_carla.py 
```
You can use --whole_scene_to_mesh to generate a complete static scene with all frames at one time, then add the moving object point cloud, and finally divide it into small scenes. In this way, we can accelerate the generation process and get denser but more uneven occupancy labels.
 

### Output and visualization
**Output**
输出物，默认在被处理数据路径下新建一个名为occ的目录，其中包括：
* `labels.npz`： 内容为字典, 值为{"semantics": semantics,  "mask_camera": mask_camera, "mask_lidar": mask_lidar}，semantics是L x W x H的occ，mask_camera与mask_lidar主要用于mask FOV之外或被遮挡的栅格。使用visual_nuscenes.py进行可视化
* `labels.npy`：N x 4的occ，可用visual.py进行可视化

**Visualization**
```
python visual.py 
```