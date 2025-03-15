import cv2
import os
import json
import time

# 图片路径和标签路径
image_folder = "./carla_data/sequences/04/image"
label_path = "./carla_data/sequences/04/labels.json"

# 帧率（每秒播放的帧数）
fps = 20
frame_delay = int(1000 / fps)  # 每帧的延迟时间（毫秒）

# 加载标签数据
with open(label_path, "r") as f:
    labels = json.load(f)

# 将标签数据按帧ID整理为字典，方便快速查找
label_dict = {str(item["frame_id"]): item["objects"] for item in labels}

# 获取图片文件列表
image_files = sorted(os.listdir(image_folder))

# 遍历图片文件并按帧率播放
for image_file in image_files:
    # 获取帧ID（假设图片文件名是帧ID，例如 "000001.png"）
    frame_id = os.path.splitext(image_file)[0]

    # 加载图片
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)

    # 如果图片加载失败，跳过
    if frame is None:
        print(f"无法加载图片: {image_path}")
        continue

    # 获取当前帧的标签数据
    if frame_id in label_dict:
        
        objects = label_dict[frame_id]
        for obj in objects:
            # 提取边界框
            bbox = obj["bbox"]
            class_name = obj["class"]

            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)  # 将浮点数转换为整数

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色框

            # 在边界框上方显示类别名称
            # cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示图片
    cv2.imshow("Frame", frame)

    # 按帧率延迟
    if cv2.waitKey(frame_delay) & 0xFF == ord("q"):  # 按 'q' 键退出
        break

# 释放资源
cv2.destroyAllWindows()