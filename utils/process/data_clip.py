import config
from utils.folder_utils import create_folders

sensor_data_dirs = [
    './image/CAM_BACK_RIGHT'
    './image/CAM_BACK_LEFT'
    './image/CAM_BACK'
    './image/CAM_FRONT'
    './image/CAM_FRONT_RIGHT'
    './image/CAM_FRONT_LEFT'
    './velodyne_semantic'
    './radar'
    './semantic'
    './depth'
    './velodyne'
]

def clip_sequence(start_frame, end_frame, input_sequence, output_sequence):
    # 创建新序列的文件夹
    main_folder = create_folders(config.DATA_DIR, output_sequence)


    # 复制原序列的文件到新序列对应的文件夹下

    # 读取csv, json等数据, 写入新的文件




if __name__ == '__main__':
    start_frame = ''
    end_frame = ''

    data_root = ''
    input_sequence = ''
    output_sequence = ''

    clip_sequence(start_frame, end_frame, input_sequence, output_sequence)

