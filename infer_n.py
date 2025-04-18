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
model_name = "/work/share/projects/mjc/lmfusion/checkpoints/new/lmfusion/t2i/checkpoint-49000"

#config = Qwen2Config.from_pretrained("/work/share/projects/mjc/lmfusion/Qwen2.5-1.5B", cache_dir='./cache_dir')

config = Qwen2Config.from_pretrained(model_name, cache_dir='./cache_dir')
    #setattr(config, '_attn_implementation_new', training_args._attn_implementation_new)

model: Qwen2ForCausalLM =Qwen2ForCausalLM.from_pretrained(
      model_name, 
   ).to("cuda")
print(config)
#model=Qwen2ForCausalLM(config).to("cuda")
model.diffusion_loss=False
# from safetensors.torch import load_file
# state_dict = load_file(os.path.join('./Qwen2.5-1.5B',"model.safetensors"))
# add_state_dict={}
# tbd_params=["q_proj","k_proj","v_proj","o_proj","mlp"]
# for key,value in state_dict.items():
#     for tbd in tbd_params:
#         if tbd in key:
#             add_state_dict[key.replace(tbd,"diffusion_"+tbd)]=deepcopy(value)
#             break
# state_dict.update(add_state_dict)

#model=model.to(dtype=torch.float16)



print(f"Model dtype: {model.dtype}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "black and white photography, fashion editorial, a dark shorthaired beautiful woman in night cafe, surrounded by enchanting bokeh lights, highres, realistic photo, professional photography, cinematic angle, dynamic light back shining, bokeh,"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
text+="<|image_start|>"
#<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant
#|im_start|:151644, <|im_end|>:151645,\n:198, ".":13
model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
model_ids = model_inputs['input_ids'].to(model.device)
#print(model.device)
print(model_ids)

from diffusers import FlowMatchEulerDiscreteScheduler     
from diffusers import SD3Transformer2DModel

scheduler=FlowMatchEulerDiscreteScheduler(**{"shift": 3.0})

num_inference_steps= 50
scheduler.set_timesteps(num_inference_steps, device=model.device)
timesteps = scheduler.timesteps

timesteps =  timesteps.to(model.dtype)

do_classifier_free_guidance  = True
output_type  = "pil"

latents = torch.randn((1, 16, 32, 32), device=model.device, dtype=model.dtype )
guidance_scale = 4
#print ("timesteps",timesteps)

with torch.no_grad():
    for i, t in enumerate(timesteps):

        timestep = t.expand(latents.shape[0])
        if do_classifier_free_guidance:
            noise_pred_text = model(
                input_ids = model_ids,
                timesteps=timestep,
                pixel_values =latents,
            )["pred"]
            noise_pred_uncond = model(
                input_ids = None,
                timesteps=timestep,
                pixel_values =latents,
            )["pred"]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = model(
                input_ids = model_ids,
                timesteps=timestep,
                pixel_values =latents,
            )["pred"]

        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if output_type == "latent":
        image = latents

    else:
        latents = (latents / model.vision_model.config.scaling_factor) + model.vision_model.config.shift_factor
        latents = latents.to(dtype=torch.float16)

        image = model.vision_model.decode(latents, return_dict=False)[0]
        vae_scale_factor = 2 ** (len(model.vision_model.config.block_out_channels) - 1) 
        image_processor = VaeImageProcessor(vae_scale_factor= vae_scale_factor)
        image = image_processor.postprocess(image.detach(), output_type=output_type)[0]

    # save_path = "/work/share/projects/mjc/lmfusion/test/img/output_image_1.png"  
    # image.save(save_path)



