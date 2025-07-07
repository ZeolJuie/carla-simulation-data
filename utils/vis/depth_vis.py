import numpy as np
import cv2

def rgb_to_depth(depth_image):
    """
    将深度相机的 RGB 图像转换为深度值（以米为单位）。
    
    :param depth_image: 深度相机的 RGB 图像（H x W x 3）。
    :return: 深度值数组（H x W）。
    """
    # 将 RGB 通道转换为深度值
    depth_meters = np.dot(depth_image[:, :, :3], [65536.0, 256.0, 1.0]) / (256 * 256 * 256 - 1)
    depth_meters *= 1000  # 转换为米
        
    return depth_meters

def depth_to_logarithmic_grayscale(depth_array, max_depth=1000):
    """
    将深度数组转换为对数深度灰度图。
    
    :param depth_array: 深度值数组（H x W）。
    :param max_depth: 深度相机的最大测量范围（默认为 1000 米）。
    :return: 对数深度灰度图（H x W）。
    """
    # 归一化深度值
    normalized_depth = depth_array / max_depth

    # 对数变换
    logdepth = np.ones(normalized_depth.shape) + (np.log(normalized_depth + 1e-6) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    logdepth *= 255.0
    return logdepth.astype(np.uint8)

def save_logarithmic_depth_image(depth_array, output_path, max_depth=1000):
    """
    将深度数组转换为对数深度灰度图并保存为图像文件。
    
    :param depth_array: 深度值数组（H x W）。
    :param output_path: 输出图像路径。
    :param max_depth: 深度相机的最大测量范围（默认为 1000 米）。
    """
    # 转换为对数深度灰度图
    log_depth_image = depth_to_logarithmic_grayscale(depth_array, max_depth)
    
    # 保存图像
    cv2.imwrite(output_path, log_depth_image)
    print(f"Logarithmic depth image saved to {output_path}")

# 示例：加载深度相机的 RGB 图像并保存为对数深度灰度图
input_path = "carla_data/sequences/09/depth/1392487.png"  # 替换为你的深度相机 RGB 图像路径
output_path = "./visualize/1392487_depth.png"  # 输出图像路径

# 加载深度相机的 RGB 图像
depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)  # 加载图像（H x W x 3）

# 将 RGB 图像转换为深度值
depth_array = rgb_to_depth(depth_image)

# 保存为对数深度灰度图
save_logarithmic_depth_image(depth_array, output_path)