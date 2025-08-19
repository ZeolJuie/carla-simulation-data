#!/bin/sh

# 定义起始和结束序列号
START_SEQ=26
END_SEQ=30

# 循环处理每个序列号
seq=$START_SEQ
while [ $seq -le $END_SEQ ]; do
    echo "正在处理序列号: $seq"
    
    # 执行校准处理
    echo "运行 calib_process.py $seq"
    if ! python utils/process/calib_process.py "$seq"; then
        echo "错误: calib_process.py 执行失败，序列号 $seq" >&2
        exit 1
    fi
    
    # 执行数据处理
    echo "运行 data_process.py --sequence $seq"
    if ! python utils/process/data_process.py --sequence "$seq"; then
        echo "错误: data_process.py 执行失败，序列号 $seq" >&2
        exit 1
    fi
    
    echo "序列号 $seq 处理完成"
    echo "----------------------------------------"
    
    seq=$((seq + 1))
done

echo "所有序列号处理完成"
exit 0