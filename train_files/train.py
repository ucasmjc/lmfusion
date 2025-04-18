import math
import copy
import json
import os
import pathlib
import random
import re
import sys
import warnings
import traceback
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch_npu.contrib import transfer_to_npu
import transformers
from dataclasses import dataclass, field
from packaging import version
from torch.utils.data import Dataset
from arg_util import ModelArguments, DataArguments, TrainingArguments
sys.path.append('./')
from train_files.dataset_v2  import  DataCollatorWithFlatteningForSupervisedDataset
#from train_files.dataset_v2  import UniDataset,
from train_files.merge_dataset import UniDataset
from trainer import JanusTrainer, safe_save_model_for_hf_trainer
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")
from qwen2 import Qwen2Config,Qwen2ForCausalLM,VLMImageProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from train_files.val_dataset import PromptDataset

def make_flattening_supervised_data_module(data_args,image_processor,tokenizer) -> Dict:
    data_args.image_under_data_files = data_args.image_under_data_files.split(',')
    data_args.image_gen_data_files = data_args.image_gen_data_files.split(',')
    data_args.text_chat_data_files = data_args.text_chat_data_files.split(',')
    data_args.sample_rate = data_args.sample_rate.split(','); data_args.sample_rate = [ int(i) for i in data_args.sample_rate]
    data_args.batchsize_list = data_args.batchsize_list.split(','); data_args.batchsize_list = [ int(i) for i in data_args.batchsize_list]

    # train_dataset = UniDataset(
    #     image_processor,
    #     tokenizer,
    #     image_under_data_files=data_args.image_under_data_files, # ['/storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json'], 
    #     image_under_rootdir=data_args.image_under_rootdir, #'/storage/yqs/dataset/BAAI/DenseFusion-1M/images',
    #     image_gen_data_files=data_args.image_gen_data_files, #['/storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json'],
    #     image_gen_rootdir=data_args.image_gen_rootdir, #'/storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded',
    #     text_chat_data_files=data_args.text_chat_data_files, #['/storage/yqs/dataset/BAAI/Infinity-Instruct/7M_domains/subjective/mini_output.json'],
    #     samples_per_epoch=data_args.samples_per_epoch, #100000,
    #     dataset=data_args.dataset, #"image_under||image_gen||text_chat",
    #     sample_rate=data_args.sample_rate, #[5, 4, 1],
    #     batchsize_list=data_args.batchsize_list #[1,1,1]
    # )
    train_dataset = UniDataset(
        "/work/share/projects/mjc/lmfusion/train_files/merge_data.json",
        image_processor=image_processor,
        tokenizer=tokenizer,
        samples_per_epoch=data_args.samples_per_epoch, #100000,
        sample_rate=data_args.sample_rate, #[5, 4, 1],
        batchsize_list=data_args.batchsize_list #[1,1,1]
    )
    
    val_dataset=PromptDataset(tokenizer)

    data_collator = DataCollatorWithFlatteningForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=None, #val_dataset,
                data_collator=data_collator)

from copy import deepcopy
from train_files.trainer import DatasetStateCallback
def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    global_rank = int(os.environ.get("RANK", "0")) 
    training_args.seed=global_rank
    # compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    compute_dtype = torch.float16 
    model_args.torch_dtype = compute_dtype
    print("model_args.torch_dtype", model_args.torch_dtype)
    config = Qwen2Config.from_pretrained(model_args.model_path, cache_dir='./cache_dir')
    if global_rank==0:
        print(training_args)
    #setattr(config, '_attn_implementation_new', training_args._attn_implementation_new)

   # model: Qwen2ForCausalLM =Qwen2ForCausalLM.from_pretrained(
    #    model_args.model_path, trust_remote_code=True, config=config,  cache_dir='./Qwen2.5-1.5B'
   # )
    model=Qwen2ForCausalLM(config)

    from safetensors.torch import load_file
    state_dict = load_file(os.path.join('./Qwen2.5-1.5B',"model.safetensors"))
    add_state_dict={}
    tbd_params=["q_proj","k_proj","v_proj","o_proj","mlp"]
    for key,value in state_dict.items():
        for tbd in tbd_params:
            if tbd in key:
                add_state_dict[key.replace(tbd,"diffusion_"+tbd)]=deepcopy(value)
                break
    state_dict.update(add_state_dict)
                
    
    m, u =model.load_state_dict(state_dict, strict=False)
    #if local_rank==1:
        #print(m,"\n")
        #print(u,"\n")
    with open(os.path.join(model_args.model_path,"preprocessor_config.json"), "r") as f:
        config_data = json.load(f)
    image_processor = VLMImageProcessor(
        **config_data
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token

    # decoupled learning rate
    model.config.diffusion_lr = training_args.diffusion_lr 

    model.requires_grad = False
    if model.config.diffusion_lr is not None:
        for n,p in model.named_parameters():
            if "diffusion" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    #if model.config.vision_encoder_lr is None:
      #  for p in model.vision_model.parameters():
       #     p.requires_grad = False
    if local_rank == 0:
        # 使用rank 0的进程写入一次
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(f'{training_args.output_dir}/trainv2_model_parameters.txt', 'w') as f:
            # 遍历所有参数及其是否需要训练
            for name, param in model.named_parameters():
                f.write(f"Parameter name: {name}\n")
                f.write(f"Requires grad: {param.requires_grad}\n")
                f.write("-" * 40 + "\n")
        print("Parameters have been logged to trainv2_model_parameters.txt")
    data_module = make_flattening_supervised_data_module(data_args=data_args,image_processor=image_processor,tokenizer=tokenizer)

    # select a Trainer
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params,trainable_params,"!!!!!!!!!!!!!!!")
    trainer = JanusTrainer(model=model, tokenizer=tokenizer, args=training_args,callbacks=[DatasetStateCallback()],  **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()