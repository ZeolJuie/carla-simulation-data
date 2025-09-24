#!/bin/bash

# 基础路径设置
BASE_DATA_PATH="/home/zmh/codes/carla-simulation-data/carla_data/sequences"
OUTPUT_BASE_PATH="/home/zmh/codes/carla-simulation-data/nuscenes_carla/gts"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_BASE_PATH"

# 要处理的序列列表（可以根据实际情况修改）
SEQUENCES=({131..164})

# 遍历所有序列并执行命令
for seq in "${SEQUENCES[@]}"; do
    # 格式化序列号为4位数（scene-0001, scene-0002等）
    seq_num=$(printf "%04d" "$((10#$seq))")
0
    # 构造完整路径
    data_path="${BASE_DATA_PATH}/${seq}"
    out_path="${OUTPUT_BASE_PATH}/scene-${seq_num}"
    
    # 创建输出目录
    mkdir -p "$out_path"
    
    # 执行命令
    echo "正在处理序列 ${seq} -> scene-${seq_num}"
    python gen_occ_walker.py --data_path "$data_path" --out_path "$out_path"
    
    # 检查命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "序列 ${seq} 处理完成"
    else
        echo "错误: 序列 ${seq} 处理失败"
        exit 1
    fi
done

echo "所有序列处理完成"