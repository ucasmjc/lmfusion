import os
import shutil

# 源路径（存放原始文件的目录）
src_dir = "/work/share1/mjc"  # 替换为你的源路径

# 目标路径（要移动到的目录）
dst_dir = "/work/share1/mjc/video_frames"  # 替换为目标路径

# 确保目标路径存在
os.makedirs(dst_dir, exist_ok=True)

# 遍历源路径下的所有文件
for filename in os.listdir(src_dir):
    # 检查文件名是否符合条件
    if filename.startswith("video_frames") and filename.endswith(".jpg"):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        # 移动文件
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")

print("移动完成！")