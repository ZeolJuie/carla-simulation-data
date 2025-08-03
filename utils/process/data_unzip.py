import os
import zipfile

def unzip_sequences_folder(base_path="carla/sequences"):
    # 检查目录是否存在
    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' not found!")
        return

    # 遍历所有ZIP文件
    for zip_filename in os.listdir(base_path):
        if not zip_filename.endswith(".zip"):
            continue

        zip_path = os.path.join(base_path, zip_filename)
        extract_folder = os.path.join(base_path, zip_filename[:-4])  # 去掉 ".zip"

        # 创建目标文件夹（如果不存在）
        os.makedirs(extract_folder, exist_ok=True)

        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_folder)

        print(f"Unzipped: {zip_filename} -> {extract_folder}")

if __name__ == "__main__":
    unzip_sequences_folder()