import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from modules.vae import WanVAE

# 初始化 WanVAE (NPU 设备)
wan_vae = WanVAE(
    z_dim=16,
    vae_pth='/work/share/projects/mjc/lmfusion/Wan2.1/ckpt/Wan2.1_VAE.pth',
    dtype=torch.float,
    device="npu"
)

def load_video_to_tensor(video_path, target_size=(256, 256), max_frames=None):
    """将 MP4 视频读取为 PyTorch 张量 (形状: [C, T, H, W])"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 读取视频帧
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    for _ in tqdm(range(total_frames), desc="Reading video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整大小并归一化到 [-1, 1]
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        frame = (frame / 127.5) - 1.0  # [0, 255] -> [-1, 1]
        frames.append(frame)
    
    cap.release()
    
    # 转换为张量并调整维度顺序
    video_tensor = torch.tensor(np.stack(frames), dtype=torch.float32)
    video_tensor = video_tensor.permute(3, 1, 0, 2)  # [T, H, W, C] -> [C, T, H, W]
    return video_tensor.to("npu")  # 移动到 NPU

def encode_video(video_path, output_z_path=None, max_frames=None):
    """编码视频并保存潜在张量"""
    # 1. 加载视频为张量
    video_tensor = load_video_to_tensor(video_path, max_frames=max_frames)
    
    # 2. 分块编码（避免内存不足）
    chunk_size = 32  # 每次处理的帧数（根据 NPU 内存调整）
    z_chunks = []
    for i in tqdm(range(0, video_tensor.shape[1], chunk_size), desc="Encoding"):
        chunk = video_tensor[:, i:i+chunk_size, :, :]
        z = wan_vae.encode([chunk])[0]  # 编码为潜在空间
        z_chunks.append(z.cpu())  # 移回 CPU 保存内存
    
    # 3. 合并结果
    z_total = torch.cat(z_chunks, dim=2)  # 沿时间维度拼接
    
    # 4. 保存结果
    if output_z_path:
        torch.save(z_total, output_z_path)
        print(f"潜在编码已保存至: {output_z_path}")
    
    return z_total

# # 示例调用
# if __name__ == "__main__":
#     video_path = "/work/share/projects/mjc/lmfusion/76gqtDiswDQ_segment_190.mp4"  # 替换为你的 MP4 路径
#     output_z_path = "/work/share/projects/mjc/lmfusion/output_z.pt"   # 潜在编码输出路径
    
#     # 编码视频
#     z = encode_video(video_path, output_z_path, max_frames=100)  # max_frames 限制帧数（可选）
#     print("潜在编码形状:", z.shape)  # 应为 [1, 16, T, H//8, W//8]




