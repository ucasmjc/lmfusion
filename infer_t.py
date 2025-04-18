import torch
from transformers import AutoTokenizer,  AutoModelForCausalLM
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, Lambda
from PIL import Image
from qwen2 import Qwen2Config,Qwen2ForCausalLM1
import numpy as np
import os
import PIL.Image,json

def preprocess(image_data: torch.Tensor, short_size: int = 256) -> torch.Tensor:
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: 2. * x - 1.), 
            Resize(size=short_size),
        ]
    )
    outputs = transform(image_data)
    outputs = outputs.unsqueeze(0)
    return outputs


model_name = "Qwen2.5-1.5B"


tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.pad_token_id) 
prompt = "Give me a short introduction to large language model."
messages = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
model=Qwen2ForCausalLM1.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(model.dtype)
print(model.device)
#https://t03itcjathb.feishu.cn/docx/SrxJdDKWSo3Gg3xY0Tvca3o7nug?from=from_copylink
model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)


import pdb 
#pdb.set_trace()

print (model_inputs)
print ("input",model_inputs.input_ids.shape )
print (model_inputs.input_ids)
print ("input_type",type(model_inputs.input_ids ))

print (model_inputs[0])

print (model_inputs.attention_mask)

image_path = "/work/share/projects/mjc/lmfusion/train_files/t2i_example/coyo1/001.jpg"
short_size = 256

image = preprocess(Image.open(image_path), short_size)
# print (image)
# print (type(image))
# print (image.shape)


generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
print(generated_ids)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)