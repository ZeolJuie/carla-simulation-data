import os
import sys
import json
import shutil

import pandas as pd

sensor_data_dirs = [
    'image/CAM_BACK_RIGHT',
    'image/CAM_BACK_LEFT',
    'image/CAM_BACK',
    'image/CAM_FRONT',
    'image/CAM_FRONT_RIGHT',
    'image/CAM_FRONT_LEFT',
    'velodyne_semantic',
    'radar',
    'semantic',
    'depth',
    'velodyne'
]

def clip_sequence(start_frame, end_frame, input_sequence, output_sequence):

    sys.path.append('.')
    from config import DATA_DIR

    # 创建新序列的文件夹
    os.makedirs(os.path.join(DATA_DIR, output_sequence), exist_ok=True)
    for sensor_data_dir in sensor_data_dirs:
        os.makedirs(os.path.join(DATA_DIR, output_sequence, sensor_data_dir), exist_ok=True)

    # 复制原序列的文件到新序列对应的文件夹下
    for sensor_data in sensor_data_dirs:
        sensor_data_dir = os.path.join(DATA_DIR, input_sequence ,sensor_data)
        output_dir = os.path.join(DATA_DIR, output_sequence ,sensor_data)
        all_files = sorted(os.listdir(sensor_data_dir))

        copy_flag = False
        for file_name in all_files:
            if start_frame in file_name:
                copy_flag = True
            if copy_flag:
                shutil.copy2(os.path.join(sensor_data_dir, file_name), output_dir)
            if end_frame in file_name:
                copy_flag = False
        
        print(f"{sensor_data} clip done!")

    # 读取csv, json等数据, 写入新的文件
    csv_files = ['ego.csv', 'gnss.csv', 'imu.csv']
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATA_DIR, input_sequence, csv_file))
        result = df[(df['frame'] >= int(start_frame)) & (df['frame'] <= int(end_frame))]
        result['frame'] = result['frame'].astype(str).str.zfill(6)
        result.to_csv(os.path.join(DATA_DIR, output_sequence, csv_file), index=False)
        print(f"{csv_file} clip done!")
        if csv_file == 'ego.csv':
            start_ts = result.iloc[0, 1]
            end_ts = result.iloc[-1, 1]
               
    labels_filename = 'labels.json'
    with open(os.path.join(DATA_DIR, input_sequence, labels_filename), 'r', encoding='utf-8') as file:
        labels = json.load(file)
        clipped_labels = [frame for frame in labels if frame["frame_id"] >= start_frame and frame["frame_id"] <= end_frame]
    
    with open(os.path.join(DATA_DIR, output_sequence, labels_filename), 'w') as f:
            json.dump(clipped_labels, f)
    print("labels.json clip done!")

    log_filename = 'log.json'
    with open(os.path.join(DATA_DIR, input_sequence, log_filename), 'r', encoding='utf-8') as file:
        log = json.load(file)
        log["scene"] = output_sequence
        log["duration"] = start_ts - end_ts
        log["frame_num"] = len(clipped_labels)
    
    with open(os.path.join(DATA_DIR, output_sequence, log_filename), 'w') as f:
            json.dump(clipped_labels, f)
    print("log.json clip done!")


if __name__ == '__main__':

    start_frame = '052340'
    end_frame = '053600'

    input_sequence = '19'
    output_sequence = '21'

    clip_sequence(start_frame, end_frame, input_sequence, output_sequence)
