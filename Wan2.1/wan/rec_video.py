import time
import random
import argparse
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
import decord

from modules.vae import WanVAE

from PIL import Image
from decord import VideoReader, cpu
from torch.nn import functional as F
from torchvision.transforms import Lambda, Compose
import sys
sys.path.append(".")


import subprocess

def check_npu_memory():
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
    print(result.stdout)

#from opensora.models.causalvideovae import ae_wrapper


#import ToTensorVideo, CenterCropResizeVideo

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:  
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]

def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    
    # import ipdb;ipdb.set_trace()
    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        # hxw 720x1280  thxtw 320x640  hw_raito 9/16 > tr_ratio 8/16  newh=1280*320/640=640  neww=1280 
        new_h = int(w * tr)
        new_w = w
    else:
        # hxw 720x1280  thxtw 480x640  hw_raito 9/16 < tr_ratio 12/16   newh=720 neww=720/(12/16)=960  
        # hxw 1080x1920  thxtw 720x1280  hw_raito 9/16 = tr_ratio 9/16   newh=1080 neww=1080/(9/16)=1920  
        new_h = h
        new_w = int(h / tr)
    
    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)

class CenterCropResizeVideo:
    '''
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    '''

    def __init__(
            self,
            size,
            top_crop=False, 
            interpolation_mode="bilinear",
    ):
        if len(size) != 2:
            raise ValueError(f"size should be tuple (height, width), instead got {size}")
        self.size = size
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_th_tw(clip, self.size[0], self.size[1], top_crop=self.top_crop)
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size,
                                         interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"



def array_to_video(image_array: npt.NDArray, fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    #x = x.permute(0, 2, 3, 1).float().numpy()
    x = x.permute(1, 2, 3, 0).float().numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    # if total_frames > sample_frames_len:
    #     s = random.randint(0, total_frames - sample_frames_len - 1)
    #     s = 0
    #     e = s + sample_frames_len
    #     num_frames = num_frames
    # else:
    # s = 0
    # e = total_frames
    # num_frames = int(total_frames / sample_frames_len * num_frames)
    s = 0
    e = sample_frames_len
    print(f'sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}', video_path,
            total_frames)

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess(video_data: torch.Tensor, height: int = 128, width: int = 128) -> torch.Tensor:
    transform = Compose(
        [
            ToTensorVideo(),
            CenterCropResizeVideo((height, width)),
            Lambda(lambda x: 2. * x - 1.)
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


def main(args: argparse.Namespace):
    device = args.device
    kwarg = {}
    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir', **kwarg).to(device)
    # vae = CausalVAEModelWrapper(args.ae_path, **kwarg).to(device)
    vae = ae_wrapper[args.ae](args.ae_path, **kwarg).eval().to(device)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        # vae.vae.tile_sample_min_size = 512
        # vae.vae.tile_latent_min_size = 64
        # vae.vae.tile_sample_min_size_t = 29
        # vae.vae.tile_latent_min_size_t = 8
        # if args.save_memory:
        #     vae.vae.tile_sample_min_size = 256
        #     vae.vae.tile_latent_min_size = 32
        #     vae.vae.tile_sample_min_size_t = 9
        #     vae.vae.tile_latent_min_size_t = 3
    dtype = torch.float16
    vae.eval()
    vae = vae.to(device, dtype=dtype)
    
    with torch.no_grad():
        x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.height,
                           args.width)
        print("input shape", x_vae.shape)
        x_vae = x_vae.to(device, dtype=dtype)  # b c t h w
        # for i in range(10000):
        latents = vae.encode(x_vae)
        latents = latents.to(dtype)
        video_recon = vae.decode(latents)  # b t c h w
        print("recon shape", video_recon.shape)


    
    # vae = vae.half()
    # from tqdm import tqdm
    # with torch.no_grad():
    #     x_vae = torch.rand(1, 3, 93, 720, 1280)
    #     print(x_vae.shape)
    #     x_vae = x_vae.to(device, dtype=torch.float16)  # b c t h w
    #     # x_vae = x_vae.to(device)  # b c t h w
    #     for i in tqdm(range(100000)):
    #         latents = vae.encode(x_vae)
    #     print(latents.shape)
    #     latents = latents.to(torch.float16)
    #     video_recon = vae.decode(latents)  # b t c h w
    #     print(video_recon.shape)


    custom_to_video(video_recon[0], fps=args.fps, output_file=args.rec_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video_path', type=str, default='')
#     parser.add_argument('--rec_path', type=str, default='')
#     parser.add_argument('--ae', type=str, default='')
#     parser.add_argument('--ae_path', type=str, default='')
#     parser.add_argument('--model_path', type=str, default='results/pretrained')
#     parser.add_argument('--fps', type=int, default=30)
#     parser.add_argument('--height', type=int, default=336)
#     parser.add_argument('--width', type=int, default=336)
#     parser.add_argument('--num_frames', type=int, default=100)
#     parser.add_argument('--sample_rate', type=int, default=1)
#     parser.add_argument('--device', type=str, default="cuda")
#     parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
#     parser.add_argument('--tile_sample_min_size', type=int, default=512)
#     parser.add_argument('--tile_sample_min_size_t', type=int, default=33)
#     parser.add_argumentx('--tile_sample_min_size_dec', type=int, default=256)
#     parser.add_argument('--tile_sample_min_size_dec_t', type=int, default=33)
#     parser.add_argument('--enable_tiling', axxction='store_true')
#     parser.add_argument('--save_memory', action='store_true')

#     args = parser.parse_args()
#     main(args)


video_path =  "/work/share/projects/mjc/lmfusion/76gqtDiswDQ_segment_190_5frames.mp4"
reader = decord.VideoReader(video_path, ctx=decord.cpu(0),num_threads=70 )
num_frames =  len(reader)
print ("num_frames", num_frames)
sample_rate = 1
height = reader[0].shape[0] 
print ("height",height)
width = reader[0].shape[1]
print ("width",width)

import time
import torch_npu
from torch_npu.contrib import transfer_to_npu

with torch.no_grad():
    start_time = time.time() 
    x_vae = preprocess(read_video(video_path, num_frames, sample_rate), height, width)
    print("input shape", x_vae.shape) 
    #  torch.Size([1, 3, 169, 720, 1280]) 01  <class 'torch.Tensor'>
    # print(x_vae)
    # print (type(x_vae))
    # print (min(x_vae))

    x_vae = x_vae.to("cuda")

    wan_vae = WanVAE(
    z_dim=16,
    vae_pth='/work/share/projects/mjc/lmfusion/Wan2.1/ckpt/Wan2.1_VAE.pth',
    dtype=torch.float16,
    device="cuda"
    )
    print(wan_vae.dtype)
    encoder_start = time.time()
    x_vae = x_vae.to(dtype = torch.float16,)

    print ("vae shape",x_vae.shape)

    print(f"CUDA 内存占用: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    latents = wan_vae.encode(x_vae)

    print ("len",len(latents))
    print ("type",type(latents[0]))
    print ("shape",latents[0].shape)

    encoder_end = time.time()
    encoder_time = encoder_end - encoder_start

    print(f"encode end CUDA 内存占用: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    video_recon = wan_vae.decode(latents)  
    print(f"CUDA 内存占用: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    print("recon type", type(video_recon))
    print ("recon shape",video_recon[0].shape)

    decoder_end = time.time()
    decoder_time = decoder_end - encoder_end

    print(f"CUDA 最大内存占用: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

    print(f"Encoder 耗时: {encoder_time:.4f} 秒")
    print(f"Decoder 耗时: {decoder_time:.4f} 秒")
    print(f"总耗时: {encoder_time + decoder_time:.4f} 秒")

    #custom_to_video(video_recon[0], fps= 30 , output_file= "/work/share/projects/mjc/lmfusion/76gqtDiswDQ_segment_190frames_10rec.mp4")


