import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import torch.distributed as dist
# from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import gc
 

import sys, os; cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cur_dir))
# import sys; sys.path.append('/storage/jp/Janus')
# import ipdb; ipdb.set_trace()

from dataset_files.image_gen_dataset import ImageGenDataset
from dataset_files.image_under_dataset import ImageUnderDataset
from dataset_files.text_chat_dataset  import TextChatDataset

from janus.utils.io import load_pil_images
from qwen2 import VLMImageProcessor

from typing import Dict, List, Optional, Sequence
import numpy as np
import json, os
import traceback
import random
import warnings
import copy
import transformers
from dataclasses import dataclass, field
import torch.distributed as dist
class DistributedRandom:
    def __init__(self):
        self.rng = random.Random()
        if dist.get_rank() == 0:
            self.rng.seed(torch.randint(0, 1000000, (1,)).item())
        self.sync_seed()

    def sync_seed(self):
        if dist.is_initialized():
            seed = torch.tensor(self.rng.getstate()[1][0]).cuda()
            dist.broadcast(seed, src=0)
            self.rng.seed(seed.item())

    def random(self):
        return self.rng.random()

# 定义一个简单的数据集
class UniDataset(Dataset):
    def __init__(self,  
                image_processor,
                tokenizer,
                image_under_data_files=None, 
                image_under_rootdir=None,
                image_gen_data_files=None,
                image_gen_rootdir=None,
                text_chat_data_files=None,
                samples_per_epoch=10000,
                dataset="image_under||image_gen||text_chat",
                sample_rate=[9, 3, 3],
                batchsize_list=[1,2,3],
                short_cap=0.2,
                ):

        self.samples_per_epoch = samples_per_epoch
        sample_rate = np.array(sample_rate)
        self.short_cap  = short_cap
        self.sample_rate = sample_rate / sample_rate.sum()
        self.batchsize_list = batchsize_list


        self.image_under_rootdir = image_under_rootdir
        self.image_gen_rootdir = image_gen_rootdir


        self.datasets = dataset.split("||")
        self.all_datasets = []
        self.all_datasets_rootdir = []
        self.image_processor=image_processor
        self.tokenizer=tokenizer
       # self.dist_rng = DistributedRandom()


        for dataset in self.datasets:
            if dataset == "image_under":
                self.all_datasets.append(
                    ImageUnderDataset(
                       image_under_data_files,
                       image_under_rootdir
                    )
                )
                self.all_datasets_rootdir.append(self.image_under_rootdir)

            elif dataset == "image_gen":
                self.all_datasets.append(
                    ImageGenDataset(
                      image_gen_data_files,
                      image_gen_rootdir
                    )
                )
                self.all_datasets_rootdir.append(self.image_gen_rootdir)


            elif dataset == "text_chat":
                self.all_datasets.append(
                    TextChatDataset(
                        text_chat_data_files,
                    )
                )
                self.all_datasets_rootdir.append('None')  
    def __len__(self):
        return  self.samples_per_epoch

    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def _convert_normal(self, data_dict, data_folder=None, short_cap=0.2):
        conversation = copy.deepcopy(data_dict["conversations"])
        # data sanity check and repair
        start_idx = 0
        for sentence in conversation:
            if sentence["from"] == "human" or sentence["from"] == "system":
                break
            start_idx += 1
        if start_idx > 0:
            warnings.warn(f"Find {start_idx} non-user sentences at the beginning of the conversation, remove them automatically!")
            conversation = conversation[start_idx:]
        if 'image_gen' in data_dict and data_dict['image_gen'] is not None:
            modal = 'image_gen'
            image_file = data_dict['image_gen']
            if isinstance(image_file, list):
                image_file = [os.path.join(self.image_gen_rootdir, f) for f in image_file]
            else:
                image_file = [os.path.join(self.image_gen_rootdir, image_file)]
        messages = []
        image_id = 0
        for conv in conversation:
            if conv["from"] == "human":
                if modal == 'image_gen':
                    if isinstance(conv["value"], list):
                        if len(conv["value"]) > 1 and random.random() <  short_cap:
                            prompt = conv["value"][1]
                        else:
                            prompt = conv["value"][0]
                    else:
                        prompt = conv["value"]
                    messages.append({
                        "role": "user",
                        "content": prompt,
                        "images": [image_file[image_id]]
                    })
                    image_id += 1
                else:
                    messages.append({
                        "role": "user",
                        "content": conv["value"],
                    })

        return modal, messages
    def _convert_imggen(self, data_dict, data_folder=None, short_cap=0.2):
        modal = 'image_gen'
        image_file = data_dict['path']
        image_file = os.path.join(self.image_gen_rootdir, image_file)
        messages = []
        if isinstance(data_dict["cap"], list):
            if len(data_dict["cap"]) > 1 and random.random() <  short_cap:
                prompt = data_dict["cap"][1]
            else:
                prompt = data_dict["cap"][0]
        else:
            prompt = data_dict["cap"]
        messages.append({
            "role": "user",
            "content": prompt,
            "images": image_file
        })

        return modal, messages
    def process_one(self,conversations, images):
        sft_format=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        #input_ids = self.tokenizer.encode(sft_format)
        #input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        #image_token_mask: torch.BoolTensor = input_ids == self.image_id
        #image_indices = image_token_mask.nonzero()
        #input_ids, num_image_tokens = self.add_image_token(
         #   image_indices=image_indices,
          #  input_ids=input_ids,
        #)

        # load images
        images_outputs = self.image_processor(images, return_tensors="pt")

        return {"sft_format":sft_format,
            "pixel_values":images_outputs.pixel_values}
            
    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        batchsize = self.batchsize_list[ind]
        data = self.all_datasets[1]

        data_batch = []
        # import ipdb; ipdb.set_trace()
        text_list=[]
        image_list=[]
        data_folder = self.image_gen_rootdir
        for b_idx in range(batchsize):
            messages = data[b_idx]
            # assert self.image_gen_rootdir == self.image_under_rootdir 
            
                #if "image_gen" in messages:
                 #   convert_func=self._convert_normal
                #else:
                    #convert_func=self._convert_imggen
            try:
                modal, messages = self._convert_imggen(
                                        messages,
                                        data_folder=data_folder, 
                                        short_cap=self.short_cap
                                        )
                # load images and prepare for inputs 
                pil_images = load_pil_images(messages)
                sft_format=self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                sft_format+="<|image_start|>"
                if random.random()< 0.1:
                    sft_format=""
                text_list.append(sft_format)
                image_list.append(self.image_processor(pil_images, return_tensors="pt").pixel_values)
            except:
                backup_idx = random.randint(0, self.__len__() - 1)
                #print(f"Encounted error when process {idx}-th example: {messages}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
 
            # data_batch.append(dict(data_dict))
        
        data_batch = {}
        res=self.tokenizer(text_list, return_tensors="pt", padding=True)
        data_batch["input_ids"]=res["input_ids"]

        
          #  print(prob,idx,dist.get_rank(),)
         #   data_batch["input_ids"] = None
        pad_mask=res["attention_mask"]
        # 扩展 pad_mask 到 (b, seq_len)，图像部分默认 1（不被 pad）
        extended_pad_mask = torch.ones((pad_mask.shape[0], pad_mask.shape[1]+256), 
                                     dtype=pad_mask.dtype, 
                                     device=pad_mask.device)
        extended_pad_mask[:, :pad_mask.shape[1]] = pad_mask 
        data_batch["pad_mask"]=extended_pad_mask.to(dtype=torch.bool) 
        #print(res["attention_mask"][:,0])
        #print(res["attention_mask"].shape,res["input_ids"].shape)
        data_batch["pixel_values"]=torch.cat(image_list,dim=0)
        data_batch['modals'] = [modal]*batchsize
        data_batch['sft_format'] = text_list
        return data_batch
import sys
@dataclass
class DataCollatorWithFlatteningForSupervisedDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = instances[0]
        #print(batch,"!!!!!!!!!!!")
        #print(batch['sft_format'][0][-60:-50],dist.get_rank())
        return batch
    

"""

# 单进程
cd /storage/jp/Janus/
python test_dataset.py


# 多进程
cd  /storage/zhubin/UniLLM
nnodes=1
nproc_per_node=2
export master_addr=127.0.0.1
export master_port=29505
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate 


HF_DATASETS_OFFLINE=1 torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
test_dataset.py  

"""