import os
import zipfile

def zip_sequences_folder(base_path="carla_data/sequences"):
    # 检查目录是否存在
    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' not found!")
        return

    # 遍历所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if folder_name not in ['01', '02']:
            continue
        
        # 跳过非文件夹的文件
        if not os.path.isdir(folder_path):
            continue

        # 构建ZIP文件名（如 "01.zip"）
        zip_filename = f"{folder_name}.zip"
        zip_path = os.path.join(base_path, zip_filename)

        # 创建ZIP文件并添加文件夹内容
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname=os.path.join(folder_name, arcname))

        print(f"Zipped: {folder_name} -> {zip_filename}")

if __name__ == "__main__":
    zip_sequences_folder()