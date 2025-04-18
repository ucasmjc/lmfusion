import torch
from transformers import AutoTokenizer,  AutoModelForCausalLM

import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import AutoencoderKL

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from diffusers import FlowMatchEulerDiscreteScheduler     
from diffusers import SD3Transformer2DModel

from qwen2 import Qwen2Config,Qwen2ForCausalLM
import numpy as np
import os
import PIL.Image,json

# 模型加载部分保持不变
model_name = "/work/share/projects/mjc/lmfusion/checkpoints/new/lmfusion/t2i/checkpoint-49000"
config = Qwen2Config.from_pretrained(model_name, cache_dir='./cache_dir')
model = Qwen2ForCausalLM.from_pretrained(model_name).to("cuda")
model.diffusion_loss = False
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 读取prompt文件
with open('/work/share/projects/mjc/lmfusion/test/prompt/prompt.txt', 'r') as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# 确保输出目录存在
output_dir = "/work/share/projects/mjc/lmfusion/test/img/cfg3.5/"
os.makedirs(output_dir, exist_ok=True)


do_classifier_free_guidance = True
output_type = "pil"
guidance_scale = 3.5

# 对每个prompt生成图像
for idx, prompt in enumerate(prompts, start=1):
    print(f"Processing prompt {idx}: {prompt}")

    scheduler = FlowMatchEulerDiscreteScheduler(**{"shift": 3.0})
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=model.device)
    timesteps = scheduler.timesteps.to(model.dtype)
    
    # 准备输入
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    text += "<|image_start|>"
    
    model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    model_ids = model_inputs['input_ids'].to(model.device)
    
    # 生成随机潜变量
    latents = torch.randn((1, 16, 32, 32), device=model.device, dtype=model.dtype)
    
    # 推理过程
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            if do_classifier_free_guidance:
                noise_pred_text = model(
                    input_ids=model_ids,
                    timesteps=timestep,
                    pixel_values=latents,
                )["pred"]
                noise_pred_uncond = model(
                    input_ids=None,
                    timesteps=timestep,
                    pixel_values=latents,
                )["pred"]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = model(
                    input_ids=model_ids,
                    timesteps=timestep,
                    pixel_values=latents,
                )["pred"]

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 后处理并保存图像
        if output_type != "latent":
            latents = (latents / model.vision_model.config.scaling_factor) + model.vision_model.config.shift_factor
            latents = latents.to(dtype=torch.float16)
            image = model.vision_model.decode(latents, return_dict=False)[0]
            vae_scale_factor = 2 ** (len(model.vision_model.config.block_out_channels) - 1)
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            image = image_processor.postprocess(image.detach(), output_type=output_type)[0]

            # 保存图像，按数字顺序命名
            save_path = os.path.join(output_dir, f"output_image_{idx}.png")
            image.save(save_path)
            print(f"Saved image to {save_path}")

print("All prompts processed!")