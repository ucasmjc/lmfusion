import torch
from transformers import AutoTokenizer,  AutoModelForCausalLM

import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import AutoencoderKL

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from qwen2 import Qwen2Config,Qwen2ForCausalLM
import numpy as np
import os
import PIL.Image,json
def sample(model,scheduler,model_ids):
    timesteps = scheduler.timesteps
    output_type  = "pil"
    latents = torch.randn((1, 16, 32, 32), device=model.device, dtype=model.dtype )
    guidance_scale = 7
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0])
        if guidance_scale>1:
            noise_pred_text = model(
                input_ids = model_ids,
                timesteps=timestep,
                pixel_values =latents,
                is_train=False
            )["pred"]
            noise_pred_uncond = model(
                input_ids = None,
                timesteps=timestep,
                pixel_values =latents,
                is_train=False
            )["pred"]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = model(
                input_ids = model_ids,
                timesteps=timestep,
                pixel_values =latents,
            )["pred"]

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = (latents / model.vision_model.config.scaling_factor) + model.vision_model.config.shift_factor
    latents = latents.to(dtype=torch.float16)

    image = model.vision_model.decode(latents, return_dict=False)[0]
    image = (image + 1) * 127.5 
    image = image.clamp(0, 255).to(torch.uint8)
    return image
