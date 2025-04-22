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
model_name = "/work/share/projects/mjc/lmfusion/checkpoints/generation/lmfusion/gen/checkpoint-250000"

model=Qwen2ForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to("cuda")
model.diffusion_loss=False

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
text+="<|vision_start|>"
# print(text)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# black and white photography, fashion editorial, a dark shorthaired beautiful woman in night cafe, surrounded by enchanting bokeh lights, highres, realistic photo, professional photography, cinematic angle, dynamic light back shining, bokeh,<|im_end|>
# <|im_start|>assistant
# <|vision_start|>
model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
model_ids = model_inputs['input_ids'].to(model.device)
print(model_ids)

from diffusers import FlowMatchEulerDiscreteScheduler     
#初始化schedule
scheduler=FlowMatchEulerDiscreteScheduler()
num_inference_steps=50
scheduler.set_timesteps(num_inference_steps, device=model.device)
timesteps = scheduler.timesteps

#因为采不出图来，cfg还没改
do_classifier_free_guidance  = False
#初始化噪声
latents = torch.randn((1, 16, 32, 32), device=model.device, dtype=model.dtype )
#guidance_scale = 4
#print ("timesteps",timesteps)
from PIL import Image
with torch.no_grad():
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0])
        if do_classifier_free_guidance:
            noise_pred_text = model(
                input_ids = model_ids,
                timesteps=timestep,
                pixel_values =latents,
                task="image_gen"
            )["pred"]
            noise_pred_uncond = model(
                input_ids = None,
                timesteps=timestep,
                pixel_values =latents,
                task="image_gen"
            )["pred"]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            #此处oise_pred为模型的直接输出img_output，在训练时的target为noise - z(z为gt latent)
            noise_pred = model(
                input_ids = model_ids,
                timesteps=timestep,
                pixel_values =latents,
                task="image_gen"
            )["pred"]

        #import pdb 
        #pdb.set_trace()
        #去噪，内部会执行，latents=latents+\delta t * noise_pred，检查过\delta t是负值，而训练时target刚好是反方向，所以合起来就是noise到z的速度没问题。
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    results=model.latent_to_image(latents).squeeze(0).permute(1, 2, 0)
    # [-1,1] -> [0,255]
    results = results.cpu()
    results = (results + 1) * 127.5  
    results = results.clamp(0, 255).to(torch.uint8)
    results= results.numpy()  # 转换为numpy
    print(results)
    print(results.shape)
    results=Image.fromarray(results)
    
    save_path = "/work/share/projects/mjc/lmfusion/test/img/output_image_1.png"  
    results.save(save_path)



