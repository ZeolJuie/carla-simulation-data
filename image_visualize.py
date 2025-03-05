import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def play_images(folder_path, delay=0.1):
    """
    连续播放指定文件夹中的图片。
    
    参数:
        folder_path (str): 包含图片的文件夹路径。
        delay (float): 每张图片显示的时间间隔（秒）。
    """
    # 获取文件夹中的所有文件
    files = sorted(os.listdir(folder_path))
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print("未找到图片文件，请检查文件夹路径！")
        return

    print(f"开始播放 {len(image_files)} 张图片，文件夹路径：{folder_path}")

    # 创建一个窗口用于显示图片
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = mpimg.imread(image_path)  # 加载图片
        ax.clear()  # 清除上一张图片
        ax.imshow(img)  # 显示图片
        ax.set_title(image_file)  # 显示图片文件名
        plt.pause(delay)  # 暂停指定时间

    plt.ioff()  # 关闭交互模式
    plt.close(fig)  # 关闭窗口
    print("播放完成！")

if __name__ == "__main__":
    folder_path = "./data/2025-03-02_14-33-09/image"  # 固定路径
    delay = 0.1  # 固定显示时间（秒）
    play_images(folder_path, delay)