import os
import time

# 定义图片存储目录
image_dir = "/work/share1/mjc/video_frames"

# 获取当前文件列表
current_files = os.listdir(image_dir)
current_count = len(current_files)

# 获取当前时间
current_time = time.time()

# 检查是否有之前的记录
if os.path.exists("/work/share/projects/mjc/lmfusion/train_files/download_progress.txt"):
    with open("/work/share/projects/mjc/lmfusion/train_files/download_progress.txt", "r") as f:
        last_time, last_count = map(float, f.read().split())
else:
    last_time, last_count = current_time, current_count

# 计算时间差和文件数量差
time_diff = current_time - last_time
count_diff = current_count - last_count

# 计算下载速度（文件/秒）
if time_diff > 0:
    download_speed = count_diff / time_diff
else:
    download_speed = 0

# 输出结果
print(f"当前文件数量: {current_count}")
print(f"新增文件数量: {count_diff}")
print(f"下载速度: {download_speed:.2f} 文件/秒")

# 更新记录
with open("/work/share/projects/mjc/lmfusion/train_files/download_progress.txt", "w") as f:
    f.write(f"{current_time} {current_count}")