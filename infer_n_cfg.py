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

# Model setup (unchanged)
model_name = "/work/share/projects/mjc/lmfusion/checkpoints/new/lmfusion/t2i/checkpoint-49000"
config = Qwen2Config.from_pretrained(model_name, cache_dir='./cache_dir')
model: Qwen2ForCausalLM =Qwen2ForCausalLM.from_pretrained(
      model_name, 
   ).to("cuda")
model.diffusion_loss = False
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt setup (unchanged)
prompt = "Sketch. Houses"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
text += "<|image_start|>"
model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
model_ids = model_inputs['input_ids'].to(model.device)




# Output directory setup
output_dir = "/work/share/projects/mjc/lmfusion/test/img/houses_cfg_comparison"
os.makedirs(output_dir, exist_ok=True)

# CFG values to test
cfg_values = [2, 3, 4, 5, 7]

for guidance_scale in cfg_values:
    print(f"Generating image with CFG scale: {guidance_scale}")
    scheduler = FlowMatchEulerDiscreteScheduler(**{"shift": 3.0})

    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=model.device)
    timesteps = scheduler.timesteps.to(model.dtype)
    
    # Initialize latents for each generation
    latents = torch.randn((1, 16, 32, 32), device=model.device, dtype=model.dtype)
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            
            # CFG implementation
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

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode and save the image
        latents = (latents / model.vision_model.config.scaling_factor) + model.vision_model.config.shift_factor
        latents = latents.to(dtype=torch.float16)
        image = model.vision_model.decode(latents, return_dict=False)[0]
        
        vae_scale_factor = 2 ** (len(model.vision_model.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        image = image_processor.postprocess(image.detach(), output_type="pil")[0]

        # Save with CFG value in filename
        save_path = os.path.join(output_dir, f"cfg{guidance_scale}.png")
        image.save(save_path)
        print(f"Saved image to: {save_path}")

print("All CFG comparisons completed!")