import json,os
import random
from torch.utils.data import IterableDataset
import json
import random
import torch
from torch.utils.data import IterableDataset
import numpy as np
from dataset_files.image_gen_dataset import ImageGenDataset
from dataset_files.image_under_dataset import ImageUnderDataset
from dataset_files.text_chat_dataset  import TextChatDataset

from janus.utils.io import load_pil_images
from qwen2 import VLMImageProcessor
import copy
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

class UniDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 image_processor=None,
                 tokenizer=None,
                 samples_per_epoch: int = 10000,
                 sample_rate: list = [1, 1, 0],
                 batchsize_list: list = [3, 3, 0],
                 short_cap: float = 0.2,
                 state_dict: Optional[dict] = None):
        super().__init__()

        self.samples_per_epoch = samples_per_epoch
        self.short_cap = short_cap
        self.sample_rate = np.array(sample_rate) / sum(sample_rate)
        self.batchsize_list = dict(zip(["image_under", "image_gen", "video_gen"], batchsize_list))
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        # 初始化数据集元数据
        self.data_meta = self._process_metadata(self.read_jsonfile(data_file))
        self.task_names = ["image_under", "image_gen", "video_gen"]

        # 初始化数据集状态
        self._init_state(state_dict)
        self._validate_datasets()
        self.boi="<|vision_start|>"
        self.eoi="<|vision_end|>"
        self.image_token="<|vision_pad|>"
        self.image_token_id=tokenizer.encode(self.image_token, return_tensors="pt")[0]
        self.boi_id=tokenizer.encode(self.boi, return_tensors="pt")[0]
        self.eoi_id=tokenizer.encode(self.eoi, return_tensors="pt")[0]
        
        self.image_id=-1000
        self.num_image_tokens=256
        self.ignore_id=-100
        self.eos_id=tokenizer.eos_token_id
        self.eos="<|endoftext|>"
        self.pad_id=tokenizer.pad_token_id
        self.generation_prompt_length=len(tokenizer("<|im_start|>assistant\n").input_ids)
        self.dist_rng = random.Random()
        self.dist_rng.seed(1234)

    def read_jsonfile(self, jsonfile: str) -> dict:
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _process_metadata(self, meta: dict) -> dict:
        """处理image_gen的目录结构"""
        processed = {"image_gen": []}
        for dataset in meta["image_gen"]:
            path, cap = dataset
            if os.path.isdir(path):
                processed["image_gen"].extend(
                    [[os.path.join(path, f), cap] 
                     for f in os.listdir(path) if f.endswith('.json')]
                )
            else:
                processed["image_gen"].append(dataset)
        return meta | processed  # Python 3.9+合并字典语法

    def _init_state(self, state_dict: Optional[dict]):
        """初始化训练状态"""
        self.dataset_pointers = {
            task: {
                "current_index": 0,
                "dataset": None,
                "iter_pos": 0,
                "completed": False
            } for task in self.task_names
        }

        if state_dict:
            self._load_state(state_dict)
        else:
            for task in self.task_names:
                self._load_next_dataset(task)

    def _load_state(self, state_dict: dict):
        """加载训练状态"""
        for task in self.task_names:
            if task in state_dict:
                pointer = state_dict[task]
                self.dataset_pointers[task].update({
                    "current_index": pointer["index"],
                    "iter_pos": pointer["iter_pos"],
                    "completed": pointer["completed"]
                })
                self._load_dataset(task, pointer["index"])

    def _validate_datasets(self):
        """验证数据集配置"""
        for task in self.task_names:
            if len(self.data_meta.get(task, [])) == 0:
                raise ValueError(f"任务 {task} 没有配置可用数据集")

    def _load_dataset(self, task: str, index: int):
        """加载指定任务的具体数据集"""
        meta = self.data_meta[task][index]
        if task == "image_under":
            self.dataset_pointers[task]["dataset"] = ImageUnderDataset(meta[0], meta[1])
        elif task == "image_gen":
            self.dataset_pointers[task]["dataset"] = ImageGenDataset(meta[0], meta[1])
        elif task == "video_gen":
            self.dataset_pointers[task]["dataset"] = TextChatDataset(meta[0])
        self.dataset_pointers[task]["current_index"] = index

    def _load_next_dataset(self, task: str) -> bool:
        """加载下一个数据集"""
        current_idx = self.dataset_pointers[task]["current_index"] + 1
        max_datasets = len(self.data_meta[task])

        if current_idx >= max_datasets:
            self.dataset_pointers[task]["completed"] = True
            return False

        self._load_dataset(task, current_idx)
        self.dataset_pointers[task]["iter_pos"] = 0
        return True

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        """核心数据获取方法"""
        task=np.random.choice(self.task_names, p=self.sample_rate)
        #task=self.dist_rng.choices(self.task_names, weights=self.sample_rate, k=1)[0]
        return self._get_batch(task)

    def _get_batch(self, task: str) -> dict:
        """获取指定任务的批量数据"""
        batch = []
        pointer = self.dataset_pointers[task]
        
        for _ in range(self.batchsize_list[task]):
            processed = self._get_single_sample(pointer,task)
            batch.append(processed)
        bastches=self._collate_batch(batch, task)
        return bastches

    def _get_single_sample(self, pointer: dict,task):
        """获取单个样本"""
        try:
            data = pointer["dataset"][pointer["iter_pos"]]
            pointer["iter_pos"] += 1
            processed = self.process_sample(data, task)
            return processed
        except IndexError:
            if not self._load_next_dataset(pointer["task"]):
                raise StopIteration
            return self._get_single_sample(pointer,task)
        except Exception as e:
            return self._get_single_sample(pointer,task)
        

    def process_sample(self, data: dict, task: str) -> dict:
        """数据处理模板方法"""
        if task == "image_under":
            return self._process_normal(data)
        elif task == "image_gen":
            return self._process_image_gen(data)
        elif task == "video_gen":
            return self._process_video_gen(data)
        return data

    def _process_normal(self, data: dict) -> dict:
        """处理图文理解数据"""
        messages = []
        for conv in data["conversations"]:
            role = "user" if conv["from"] == "human" else "assistant"
            content = conv["value"].replace("<image>", self.image_token)
            
            message = {"role": role, "content": content}
            if role == "user" and "<image>" in conv["value"]:
                message["images"] = data["image"]
            
            messages.append(message)
        images=load_pil_images(messages)

        training_input_ids_list = []
        targets_list = []
        
        for message_idx, message in enumerate(messages):
            if message_idx == 0:
                prompt = self.tokenizer.apply_chat_template([message], tokenize=False)
                sft_format=prompt
            else:
                if message["role"] == "assistant":
                    prompt = "<|im_start|>"+message["role"] + "\n" + message["content"]+self.eos+"\n"
                else:
                    prompt = "<|im_start|>"+message["role"] + "\n" + message["content"]+"<|im_end|>"+"\n"
                sft_format=sft_format+prompt
            training_input_ids = self.tokenizer.encode(prompt,max_length=512, truncation=True,return_tensors="pt")[0]

            training_input_ids_list.append(training_input_ids)
            # 不计算 loss  输入的东西 都不参与loss 运算

            targets = torch.full_like(training_input_ids, self.ignore_id)
            if message["role"] == "assistant":
                targets[self.generation_prompt_length:] = training_input_ids[self.generation_prompt_length:].clone()

            targets_list.append(targets)

        targets = torch.cat(targets_list)
        training_input_ids = torch.cat(training_input_ids_list)
        
        training_input_ids,targets=self.add_image_token(training_input_ids,targets)
        return {
            "text": sft_format,
            "images": images,
            "input_ids":training_input_ids,
            "labels":targets
        }
        
    def _process_image_gen(self, data: dict) -> dict:
        """处理图像生成数据"""
        if isinstance(data["cap"], list):
            if len(data["cap"]) > 1 and random.random() <  short_cap:
                prompt = data["cap"][1]
            else:
                prompt = data["cap"][0]
        else:
            prompt = data["cap"]
        sft_format=self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,add_generation_prompt=True) + self.boi
        if random.random()< 0.1:
            sft_format=""
        return {
            "text": sft_format,
            "images": load_pil_images([{"images": data["path"]}])
        }
    def add_image_token(
        self,
        input_ids: torch.LongTensor,
        targets
    ):  
        #import pdb; pdb.set_trace()
        image_token_mask : torch.BoolTensor = input_ids == self.image_token_id
        image_indices = image_token_mask.nonzero()
        input_slices = []
        targets_slices = []
        start = 0
        for index in image_indices:
            end = index + 1
            # original text tokens
            input_slices.append(input_ids[start:end])

            # add boi, image tokens, eoi and set the mask as False
            input_slices.append(self.boi_id * torch.ones((1), dtype=torch.long))
            input_slices.append(
                self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )
            input_slices.append(self.eoi_id * torch.ones((1), dtype=torch.long))
            if targets is not None:
                targets_slices.append(targets[start:end])
                targets_slices.append(self.ignore_id * torch.ones((1), dtype=torch.long))
                targets_slices.append(
                    self.ignore_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
                )
                targets_slices.append(self.ignore_id * torch.ones((1), dtype=torch.long))
            start = index + 1 
            # 图片理解 -1000 站位。  256  

        input_slices.append(input_ids[start:])
        
        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))
        if targets is not None:
            targets_slices.append(targets[start:])
            targets = torch.cat(targets_slices, dim=0)
        return input_ids, targets
  
    def _collate_batch(self, batch: list, task: str) -> dict:
        """批量数据处理"""
        
        images = [self.image_processor(item["images"]) for item in batch]
        
        pixel_values = torch.cat([img.pixel_values for img in images], dim=0)
        if task == "image_under":
            seq_lens = []
            batch_size=len(batch)
            for item in batch:
                seq_lens.append(len(item["input_ids"]))
            input_token_max_len = max(seq_lens)
            batched_input_ids = torch.full(
                (batch_size, input_token_max_len), self.pad_id
            ).long() 
            batched_labels_ids = torch.full(
                (batch_size, input_token_max_len), self.ignore_id
            ).long() 
            # 默认都不计算损失 初始化
            sft_format=[]

            batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
            batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
            for i,item in enumerate(batch):
                input_ids =item["input_ids"]
                seq_len = len(input_ids)
                batched_attention_mask[i, -seq_len:] = 1
                batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
                batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id

                sft_format.append(item["text"])
                labels = item["labels"]
                batched_labels_ids[i, -seq_len:] = torch.LongTensor(labels)
            import pdb 
            pdb.set_trace()
            return {
                "input_ids": batched_input_ids,
                "pad_mask": batched_attention_mask,
                "pixel_values": pixel_values,
                "task": task,
                "sft_format":sft_format,
                "labels":batched_labels_ids,
                "image_mask":batched_images_seq_mask
            }
            #
        elif task == "image_gen":
            seq_lens = []
            batch_size=len(batch)
            for item in batch:
                item["input_ids"]=self.tokenizer.encode(item["text"],max_length=512, truncation=True, return_tensors="pt")[0]
                seq_lens.append(len(item["input_ids"]))
            input_token_max_len = max(seq_lens)
            batched_input_ids = torch.full(
                (batch_size, input_token_max_len), self.pad_id
            ).long() 
            sft_format=[]
            batched_attention_mask = torch.zeros((batch_size, input_token_max_len+self.num_image_tokens)).bool()
            for i,item in enumerate(batch):
                input_ids =item["input_ids"]
                seq_len = len(input_ids)
                batched_attention_mask[i, -seq_len-self.num_image_tokens:] = True
                if seq_len!=0:
                    batched_input_ids[i, -seq_len:] = input_ids
                sft_format.append(item["text"])
            return {
                "input_ids": batched_input_ids,
                "pad_mask": batched_attention_mask,
                "pixel_values": pixel_values,
                "task": task,
                "sft_format":sft_format,
            }

    def state_dict(self):
        return {
            "task_pointers": {
                task: {
                    "current_index": self.dataset_pointers[task]["current_index"],
                    "iter_pos": self.dataset_pointers[task]["iter_pos"],
                    "completed": self.dataset_pointers[task]["completed"]
                } for task in self.task_names
            }
        }
    
    def load_state_dict(self, state_dict):
        for task in self.task_names:
            if task in state_dict["task_pointers"]:
                self.dataset_pointers[task].update(state_dict["task_pointers"][task])
                self._load_dataset(
                    task=task,
                    index=state_dict["task_pointers"][task]["current_index"]
                )
                # 设置数据集读取位置
                self.dataset_pointers[task]["iter_pos"] = state_dict["task_pointers"][task]["iter_pos"]
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
if __name__=="__main__":
    model_path="/work/share/projects/mjc/lmfusion/Qwen2.5-1.5B"
    with open(os.path.join(model_path,"preprocessor_config.json"), "r") as f:
        config_data = json.load(f)
    image_processor = VLMImageProcessor(
        **config_data
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #print(tokenizer(["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis image is a collage of four different scenes, each showcasing various food items and decorations. The top left panel features a bottle of Hershey's Simply Chocolate syrup surrounded by plates of chocolate eclairs and bowls of strawberries. There are also glasses filled with cherries in red liquid. The top right panel shows bananas and a bowl of cherries in red liquid, similar to the ones in the other panels. The bottom left panel displays bowls of colorful sprinkles and glasses filled with cherries in red liquid, with yellow flowers in the background. The bottom right panel shows a plate of chocolate eclairs with bananas in the background. The overall aesthetic is vibrant and festive, with a focus on sweet treats and colorful decorations. The medium appears to be a photo, with a clear and realistic depiction of the food and objects.<|im_end|>\n<|im_start|>assistant\n<|vision_start|>", ''], padding=True, return_tensors="pt"))
    dataset=UniDataset("/work/share/projects/mjc/lmfusion/train_files/merge_data.json",batchsize_list=[1, 1, 0],sample_rate= [1, 3, 0],tokenizer=tokenizer,image_processor=image_processor)
    state_dict = torch.load("dataset_state.pt")
    dataset.load_state_dict(state_dict)
    print(dataset.dataset_pointers)
    for i, data in enumerate(dataset):
        print(data["task"])
