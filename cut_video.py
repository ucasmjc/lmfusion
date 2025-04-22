import cv2

def extract_frames(input_video_path, output_video_path, num_frames=10):
    """
    从视频中提取指定数量的帧并保存为新视频
    
    参数:
        input_video_path (str): 输入视频文件路径
        output_video_path (str): 输出视频文件路径
        num_frames (int): 要提取的帧数，默认为10
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    
    
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 检查请求的帧数是否超过视频总帧数
    if num_frames > total_frames:
        print(f"警告: 视频只有 {total_frames} 帧，少于请求的 {num_frames} 帧")
        num_frames = total_frames
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 写入帧到输出视频
        out.write(frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"成功提取前{frame_count}帧并保存到 {output_video_path}")

# 使用示例
input_video = "input.mp4"  # 替换为你的输入视频路径
output_video = "output.mp4"  # 输出视频路径

# 提取前10帧(使用默认值)
extract_frames(input_video, output_video)

# 或者提取前25帧
# extract_frames(input_video, output_video, num_frames=25)

# 使用示例
input_video = "/work/share/projects/mjc/lmfusion/76gqtDiswDQ_segment_190.mp4"  # 替换为你的输入视频路径
output_video = "/work/share/projects/mjc/lmfusion/76gqtDiswDQ_segment_190_10frames.mp4" # 输出视频路径
extract_frames(input_video, output_video,num_frames=10)
