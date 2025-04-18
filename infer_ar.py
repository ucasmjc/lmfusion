import torch
from transformers import AutoTokenizer,  AutoModelForCausalLM
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, Lambda
from PIL import Image
from qwen2 import Qwen2Config,Qwen2ForCausalLM
from qwen2 import VLMImageProcessor

import numpy as np
import os
import PIL.Image,json
import torch_npu
from torch_npu.contrib import transfer_to_npu
# def preprocess(image_data: torch.Tensor, ) -> torch.Tensor:
#     transform = Compose(
#         [
#             ToTensor(),
#             Lambda(lambda x: 2. * x - 1.), 
#             Resize(size=(256, 256)), 
#         ]
#     )
#     outputs = transform(image_data)
#     outputs = outputs.unsqueeze(0)
#     return outputs

#model_path="/work/share/projects/mjc/lmfusion/Qwen2.5-1.5B"

with torch.no_grad():
    with open("/work/share/projects/mjc/lmfusion/Qwen2.5-1.5B/preprocessor_config.json", "r") as f:
        config_data = json.load(f)
    image_processor = VLMImageProcessor(
        **config_data
    )

    model_name = "Qwen2.5-1.5B"

    config = Qwen2Config.from_pretrained("/work/share/projects/mjc/lmfusion/Qwen2.5-1.5B", cache_dir='./cache_dir')
   # model=Qwen2ForCausalLM(config).to("cuda")

    model = Qwen2ForCausalLM(config).to("cuda", dtype=torch.float16)
    model.diffusion_loss=False
    model.ar_loss=False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "<|vision_pad|>What are the colors of the bus in the image?"
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)

    model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    # input_len = len (model_inputs['input_ids'])
    image_token_id=tokenizer.encode("<|vision_pad|>", return_tensors="pt")[0].to(model.device)
    boi_id=tokenizer.encode("<|vision_start|>", return_tensors="pt")[0].to(model.device)
    eoi_id=tokenizer.encode("<|vision_end|>", return_tensors="pt")[0].to(model.device)
    print ("image_token_id",image_token_id)
    image_id = -1000
    num_image_tokens =256




    input_ids = model_inputs['input_ids'][0]  # 获取 input_ids 张量
    print (type(input_ids))
    print (input_ids)
    print (input_ids.shape)

  

    image_token_mask : torch.BoolTensor = input_ids ==  image_token_id
    image_indices = image_token_mask.nonzero()
    input_slices = []
    targets_slices = []
    start = 0
    for index in image_indices:
        end = index + 1
        # original text tokens
        input_slices.append(input_ids[start:end])

        # add boi, image tokens, eoi and set the mask as False
        input_slices.append(boi_id * torch.ones((1), dtype=torch.long,device=model.device))
        input_slices.append(
            image_id * torch.ones((num_image_tokens,), dtype=torch.long, device=model.device)
        )
        input_slices.append(eoi_id * torch.ones((1), dtype=torch.long, device=model.device))
        # if targets is not None:
        #     targets_slices.append(targets[start:end])
        #     targets_slices.append(self.ignore_id * torch.ones((1), dtype=torch.long))
        #     targets_slices.append(
        #         self.ignore_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
        #     )
        #     targets_slices.append(self.ignore_id * torch.ones((1), dtype=torch.long))
        start = index + 1 
        # 图片理解 -1000 站位。  256  

    input_slices.append(input_ids[start:])
    
    # concat all slices
    input_ids = torch.cat(input_slices, dim=0)
    input_ids = input_ids.to(model.device)
    sequence_length = input_ids.shape[0]  # 获取序列长度（第1维是 batch_size，第2维是序列长度）


    image_path = "/work/share/projects/mjc/lmfusion/train_files/t2i_example/coyo1/001.jpg"
    # short_size = 128

    print(f"Model dtype: {model.dtype}")

    image = Image.open(image_path)
    processed_image = image_processor([image])  # 注意这里用列表包裹

    print (processed_image)
    print (type(processed_image))
    # print (image.shape)
    #image = image.to(model.device)
    # 确保 image tensor 与模型在同一设备上，并且数据类型一致
    processed_image  = processed_image.to(device=model.device, dtype=model.dtype)


    # images_seq_mask = torch.zeros((1,sequence_length)).long()
    images_seq_mask = torch.zeros(
            (1, sequence_length), 
            dtype=torch.bool, 
            device=model.device  # 关键：指定设备为 model.device
        )
    images_seq_mask[0:] = input_ids == image_id  # self.image_id

    print ("imglen",len(images_seq_mask))
    print (images_seq_mask)

    pixel_values = processed_image.pixel_values
    # print(model_inputs)

    print (pixel_values)
    print (type(pixel_values))
    print (pixel_values.shape)
    import pdb 
    #pdb.set_trace()

    input_ids = input_ids.unsqueeze(0)
    attention_mask = (input_ids != 0).long().to(model.device) 

    generated_ids = model.generate(
        input_ids = input_ids,
        #attention_mask = attention_mask,
        task="image_under",
        pixel_values = pixel_values,
        image_mask = images_seq_mask,
        max_new_tokens=512,
    #    model_kwargs={"task": "image_under"}, 
    )
    print(generated_ids)
    generated_ids = [
        opt_ids[len(ipt_ids):] for ipt_ids, opt_ids in zip(input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)