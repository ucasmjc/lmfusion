# Modified from Latte: https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py

import torch
import random
import numbers
from torchvision.transforms import RandomCrop, RandomResizedCrop
import statistics
import numpy as np
import ftfy
import regex as re
import html
from PIL import Image

from math import ceil, floor

from mindspeed_mm.data.data_utils.aspect_ratio import get_closest_ratio

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("Clip should be Tensor, but it is %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("Clip should be 4D, but it is %dD" % clip.dim())

    return True


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=True, antialias=True)


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = clip.size(-2), clip.size(-1)
    scale_ = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode, align_corners=True, antialias=True)



def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip

def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)

def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)

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

def random_shift_crop(clip):
    '''
    Slide along the long edge, with the short edge as crop size
    '''
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)

    if h <= w:
        long_edge = w
        short_edge = h
    else:
        long_edge = h
        short_edge = w

    th, tw = short_edge, short_edge

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return crop(clip, i, j, th, tw)


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
        raise TypeError(
            "Clip tensor should have data type uint8, but it is %s" % str(clip.dtype)
        )
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def to_tensor_after_resize(clip):
    """
    Convert resized tensor to [0, 1]
    Args:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W), but in [0, 1]
    """
    _is_tensor_video_clip(clip)
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0

def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    # print(mean)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


class RandomCropVideo:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: randomly cropped video clip.
                size is (T, C, OH, OW)
        """
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
        
TRAINING_RAITOS = {
    "0.56": (9, 16),
    "0.75": (3, 4),
    "1.00": (1, 1),
    "1.33": (4, 3),
    "1.78": (16, 9),
}
    
def find_closest_ratio(h, w):
    return TRAINING_RAITOS[get_closest_ratio(h, w, TRAINING_RAITOS)]

def get_params(h, w, stride, force_5_ratio=True):

    if not force_5_ratio:
        th, tw = h // stride * stride, w // stride * stride
        i = (h - th) // 2
        j = (w - tw) // 2
    else:
        ratio_h, ratio_w = find_closest_ratio(h, w)
        if h / w > ratio_h / ratio_w:
            target_h, target_w = int(w * ratio_h / ratio_w), w
        else:
            target_h, target_w = h, int(h * ratio_w / ratio_h)
        # center crop
        i, j = (h - target_h) // 2, (w - target_w) // 2
        # stride crop
        th, tw = target_h // stride * stride, target_w // stride * stride
        i = (target_h - th) // 2 + i
        j = (target_w - tw) // 2 + j

    return i, j, th, tw 

class SpatialStrideCropVideo:
    def __init__(self, stride, force_5_ratio=True):
        self.stride = stride
        self.force_5_ratio = force_5_ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: cropped video clip by stride.
                size is (T, C, OH, OW)
        """
        h, w = clip.shape[-2:] 
        i, j, h, w = get_params(h, w, self.stride, self.force_5_ratio)
        return crop(clip, i, j, h, w)


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stride={self.stride})"  

def get_hw_for_ratio(ratio_h, ratio_w, max_hxw, hw_stride):
    k = floor(np.sqrt(max_hxw / (ratio_h * ratio_w * hw_stride * hw_stride)))
    if k > 0:
        h = int(ratio_h * k * hw_stride)
        w = int(ratio_w * k * hw_stride)
    else:
        k = np.sqrt(max_hxw / (ratio_h * ratio_w))
        h = int(ratio_h * k)
        w = int(ratio_w * k)
    return h, w

def shortsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w
    
    if h / w > size[0] / size[1]:
        # hxw 720x1280  size 320x640  hw_raito 9/16 > size_ratio 8/16  neww=320/720*1280=568  newh=320  
        h = int(size[1] / w * h)
        w = size[1]
    else:
        # hxw 720x1280  size 480x640  hw_raito 9/16 < size_ratio 12/16   newh=640/1280*720=360 neww=640  
        # hxw 1080x1920  size 720x1280  hw_raito 9/16 = size_ratio 9/16   newh=1280/1920*1080=720 neww=1280  
        w = int(size[0] / h * w)
        h = size[0]
    return h, w


def longsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w
    
    if h / w > size[0] / size[1]:
        # hxw 720x1280  size 320x640  hw_raito 9/16 > size_ratio 8/16  neww=320/720*1280=568  newh=320  
        w = int(size[0] / h * w)
        h = size[0]
    else:
        # hxw 720x1280  size 480x640  hw_raito 9/16 < size_ratio 12/16   newh=640/1280*720=360 neww=640  
        # hxw 1080x1920  size 720x1280  hw_raito 9/16 = size_ratio 9/16   newh=1280/1920*1080=720 neww=1280  
        h = int(size[1] / w * h)
        w = size[1]
    return h, w

def maxhwresize(ori_height, ori_width, max_hxw, force_5_ratio=True, hw_stride=16):

    if ori_height * ori_width > max_hxw:
        if force_5_ratio:
            ratio_h, ratio_w = find_closest_ratio(ori_height, ori_width)
            target_h, target_w = get_hw_for_ratio(ratio_h, ratio_w, max_hxw, hw_stride)
            new_height, new_width = shortsideresize(ori_height, ori_width, (target_h, target_w), skip_low_resolution=False)
        else:
            scale_factor = np.sqrt(max_hxw / (ori_height * ori_width))
            new_height = int(ori_height * scale_factor)
            new_width = int(ori_width * scale_factor)
    else:
        if force_5_ratio:
            ratio_h, ratio_w = find_closest_ratio(ori_height, ori_width)
            target_h, target_w = get_hw_for_ratio(ratio_h, ratio_w, ori_height * ori_width, hw_stride)
            new_height, new_width = shortsideresize(ori_height, ori_width, (target_h, target_w), skip_low_resolution=False)
        else:
            new_height = ori_height
            new_width = ori_width
    return new_height, new_width

class LongSideResizeVideo:
    '''
    First use the long side,
    then resize to the specified size
    '''

    def __init__(
            self,
            size,
            skip_low_resolution=False, 
            interpolation_mode="bilinear",
    ):
        self.size = size
        self.skip_low_resolution = skip_low_resolution
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = longsideresize(h, w, self.size, self.skip_low_resolution)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(clip, target_size=(tr_h, tr_w),
                                         interpolation_mode=self.interpolation_mode)
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class MaxHWStrideResizeVideo:
    '''
    First use the h*w,
    then resize to the specified size
    '''

    def __init__(
            self,
            max_hxw,
            interpolation_mode="bilinear",
            force_5_ratio=True,
            hw_stride=16,
    ):
        self.max_hxw = max_hxw
        self.interpolation_mode = interpolation_mode
        self.force_5_ratio = force_5_ratio
        self.hw_stride = hw_stride

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = maxhwresize(h, w, self.max_hxw, force_5_ratio=self.force_5_ratio, hw_stride=self.hw_stride)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(clip, target_size=(tr_h, tr_w),
                                         interpolation_mode=self.interpolation_mode)
        
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"

class MaxHWResizeVideo:
    '''
    First use the h*w,
    then resize to the specified size
    '''

    def __init__(
            self,
            max_hxw,
            interpolation_mode="bilinear",
    ):
        self.max_hxw = max_hxw
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = maxhwresize(h, w, self.max_hxw)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(clip, target_size=(tr_h, tr_w),
                                         interpolation_mode=self.interpolation_mode)
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"

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

class CenterCropVideo:
    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip must be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


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
    

class ToTensorAfterResize:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W), but in [0, 1]
        """
        return to_tensor_after_resize(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"



def calculate_statistics(data):
    if len(data) == 0:
        return None
    data = np.array(data)
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    minimum = np.min(data)
    maximum = np.max(data)

    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': minimum,
        'max': maximum
    }


class AENorm:
    """
    Apply an ae_norm to a PIL image or video.
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Apply the center crop to the input video.

        Args:
            video (clip): The input video.

        Returns:
            video: The ae_norm video.
        """

        clip = 2.0 * clip - 1.0
        return clip

    def __repr__(self) -> str:
        return self.__class__.__name__


#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index

class DynamicSampleDuration(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, t_stride, extra_1):
        self.t_stride = t_stride
        self.extra_1 = extra_1

    def __call__(self, t, h, w):
        if self.extra_1:
            t = t - 1
        truncate_t_list = list(range(t+1))[t//2:][::self.t_stride]  # need half at least
        truncate_t = random.choice(truncate_t_list)
        if self.extra_1:
            truncate_t = truncate_t + 1
        return 0, truncate_t

class Expand2Square:
    """
    Expand the given PIL image to a square by padding it with a background color.
    Args:
        mean (sequence): Sequence of means for each channel.
    """

    def __init__(self, mean):
        self.background_color = tuple(int(x * 255) for x in mean)

    def __call__(self, pil_img):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), self.background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), self.background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class JpegDegradationSimulator:
    """
    Degrade an image based on the JPEG quality.
    """

    def __init__(self, min_quality=75, max_quality=101):
        """
        Initialize the simulator with a list of qualities.
        """
        self.qualities = list(range(min_quality, max_quality))
        self.jpeg_degrade_functions = {
            quality: self._simulate_jpeg_degradation(quality) for quality in self.qualities
        }

    def _simulate_jpeg_degradation(self, quality):
        """
        Create a function to degrade an image based on the JPEG quality.
        """

        def jpeg_degrade(img):
            with io.BytesIO() as output:
                img.convert("RGB").save(output, format="JPEG", quality=quality)
                output.seek(0)  # Move the reading cursor to the start of the stream
                img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
            return img_jpeg

        return jpeg_degrade

    def __call__(self, img):
        """
        Apply a random JPEG degradation from the available qualities.
        """
        transform = T.RandomChoice([T.Lambda(self.jpeg_degrade_functions[quality]) for quality in self.qualities])
        return transform(img)

    def __repr__(self):
        """
        Represent the class instance.
        """
        return f"JpegDegradationSimulator(qualities={self.qualities})"


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        if not all(
                mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ):
            raise Exception(f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}")
        if not all(
                mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ):
            raise Exception(f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}")
        if not all(
                mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ):
            raise Exception(f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}")
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        if not math.isclose(
                sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ):
            raise Exception(f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}")
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        vae_micro_frame_size = 17
        video_frames = x.shape[1]
        temporal_vae_downsample = 4
        num_frames = (video_frames // vae_micro_frame_size) * \
                     math.ceil(vae_micro_frame_size / temporal_vae_downsample)
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos: random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos: random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask


keywords = [
        ' man ', ' woman ', ' person ', ' people ', 'human',
        ' individual ', ' child ', ' kid ', ' girl ', ' boy ',
    ]
keywords += [i[:-1] + 's ' for i in keywords]

masking_notices = [
    "Note: The faces in this image are blurred.",
    "This image contains faces that have been pixelated.",
    "Notice: Faces in this image are masked.",
    "Please be aware that the faces in this image are obscured.",
    "The faces in this image are hidden.",
    "This is an image with blurred faces.",
    "The faces in this image have been processed.",
    "Attention: Faces in this image are not visible.",
    "The faces in this image are partially blurred.",
    "This image has masked faces.",
    "Notice: The faces in this picture have been altered.",
    "This is a picture with obscured faces.",
    "The faces in this image are pixelated.",
    "Please note, the faces in this image have been blurred.",
    "The faces in this photo are hidden.",
    "The faces in this picture have been masked.",
    "Note: The faces in this picture are altered.",
    "This is an image where faces are not clear.",
    "Faces in this image have been obscured.",
    "This picture contains masked faces.",
    "The faces in this image are processed.",
    "The faces in this picture are not visible.",
    "Please be aware, the faces in this photo are pixelated.",
    "The faces in this picture have been blurred.", 
]

webvid_watermark_notices = [
    "This video has a faint Shutterstock watermark in the center.", 
    "There is a slight Shutterstock watermark in the middle of this video.", 
    "The video contains a subtle Shutterstock watermark in the center.", 
    "This video features a light Shutterstock watermark at its center.", 
    "A faint Shutterstock watermark is present in the middle of this video.", 
    "There is a mild Shutterstock watermark at the center of this video.", 
    "This video has a slight Shutterstock watermark in the middle.", 
    "You can see a faint Shutterstock watermark in the center of this video.", 
    "A subtle Shutterstock watermark appears in the middle of this video.", 
    "This video includes a light Shutterstock watermark at its center.", 
]


high_aesthetic_score_notices_video = [
    "This video has a high aesthetic quality.", 
    "The beauty of this video is exceptional.", 
    "This video scores high in aesthetic value.", 
    "With its harmonious colors and balanced composition.", 
    "This video ranks highly for aesthetic quality", 
    "The artistic quality of this video is excellent.", 
    "This video is rated high for beauty.", 
    "The aesthetic quality of this video is impressive.", 
    "This video has a top aesthetic score.", 
    "The visual appeal of this video is outstanding.", 
]

low_aesthetic_score_notices_video = [
    "This video has a low aesthetic quality.", 
    "The beauty of this video is minimal.", 
    "This video scores low in aesthetic appeal.", 
    "The aesthetic quality of this video is below average.", 
    "This video ranks low for beauty.", 
    "The artistic quality of this video is lacking.", 
    "This video has a low score for aesthetic value.", 
    "The visual appeal of this video is low.", 
    "This video is rated low for beauty.", 
    "The aesthetic quality of this video is poor.", 
]


high_aesthetic_score_notices_image = [
    "This image has a high aesthetic quality.", 
    "The beauty of this image is exceptional", 
    "This photo scores high in aesthetic value.", 
    "With its harmonious colors and balanced composition.", 
    "This image ranks highly for aesthetic quality.", 
    "The artistic quality of this photo is excellent.", 
    "This image is rated high for beauty.", 
    "The aesthetic quality of this image is impressive.", 
    "This photo has a top aesthetic score.", 
    "The visual appeal of this image is outstanding.", 
]

low_aesthetic_score_notices_image = [
    "This image has a low aesthetic quality.", 
    "The beauty of this image is minimal.", 
    "This image scores low in aesthetic appeal.", 
    "The aesthetic quality of this image is below average.", 
    "This image ranks low for beauty.", 
    "The artistic quality of this image is lacking.", 
    "This image has a low score for aesthetic value.", 
    "The visual appeal of this image is low.", 
    "This image is rated low for beauty.", 
    "The aesthetic quality of this image is poor.", 
]

high_aesthetic_score_notices_image_human = [
    "High-quality image with visible human features and high aesthetic score.", 
    "Clear depiction of an individual in a high-quality image with top aesthetics.", 
    "High-resolution photo showcasing visible human details and high beauty rating.", 
    "Detailed, high-quality image with well-defined human subject and strong aesthetic appeal.", 
    "Sharp, high-quality portrait with clear human features and high aesthetic value.", 
    "High-quality image featuring a well-defined human presence and exceptional aesthetics.", 
    "Visible human details in a high-resolution photo with a high aesthetic score.", 
    "Clear, high-quality image with prominent human subject and superior aesthetic rating.", 
    "High-quality photo capturing a visible human with excellent aesthetics.", 
    "Detailed, high-quality image of a human with high visual appeal and aesthetic value.", 
]


def add_masking_notice(caption):
    if any(keyword in caption for keyword in keywords):
        notice = random.choice(masking_notices)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption

def add_webvid_watermark_notice(caption):
    notice = random.choice(webvid_watermark_notices)
    return random.choice([caption + ' ' + notice, notice + ' ' + caption])

def add_aesthetic_notice_video(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_video)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_video)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption



def add_aesthetic_notice_image(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_image)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_image)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption

def add_high_aesthetic_notice_image(caption):
    notice = random.choice(high_aesthetic_score_notices_image)
    return random.choice([caption + ' ' + notice, notice + ' ' + caption])

def add_high_aesthetic_notice_image_human(caption):
    notice = random.choice(high_aesthetic_score_notices_image_human)
    return random.choice([caption + ' ' + notice, notice + ' ' + caption])

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def clean_youtube(text, is_tags=False):
    text = text.lower() + ' '
    text = re.sub(
        r'#video|video|#shorts|shorts| shorts|#short| short|#youtubeshorts|youtubeshorts|#youtube| youtube|#shortsyoutube|#ytshorts|ytshorts|#ytshort|#shortvideo|shortvideo|#shortsfeed|#tiktok|tiktok|#tiktokchallenge|#myfirstshorts|#myfirstshort|#viral|viralvideo|viral|#viralshorts|#virlshort|#ytviralshorts|#instagram',
        ' ', text)
    text = re.sub(r' s |short|youtube|virlshort|#', ' ', text)
    pattern = r'[^a-zA-Z0-9\s\.,;:?!\'\"|]'
    if is_tags:
        pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    text = whitespace_clean(basic_clean(text))
    return text

def clean_vidal(text):
    title_hashtags = text.split('#')
    title, hashtags = title_hashtags[0], '#' + '#'.join(title_hashtags[1:])
    title = clean_youtube(title)
    hashtags = clean_youtube(hashtags, is_tags=True)
    text = title + ', ' + hashtags
    if text == '' or text.isspace():
        raise ValueError('text is empty')
    return text


if __name__ == '__main__':

    from torchvision import transforms
    import torch

    max_hxw = 256 * 256
    hw_stride = 16

    transform = transforms.Compose([
        MaxHWStrideResizeVideo(max_hxw=max_hxw, force_5_ratio=True, hw_stride=hw_stride), 
        SpatialStrideCropVideo(stride=hw_stride, force_5_ratio=True), 
    ])

    image = torch.randn([1, 3, 352, 640])
    print(transform(image).shape)
