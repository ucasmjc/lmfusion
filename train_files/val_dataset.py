import torch
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self,tokenizer, file_path="/work/share/projects/mjc/lmfusion/test/prompt/prompt.txt"):
        self.prompts = []
        self.tokenizer=tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                prompt = line.strip()
                if prompt:  # 确保不是空行
                    self.prompts.append(prompt)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        messages=[{
            "role": "user",
            "content": self.prompts[idx],
        }]
        sft_format=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        data_batch={}
        sft_format+="<|image_start|>"
        text_list=[sft_format]
        res=self.tokenizer(text_list, return_tensors="pt", padding=True)
        data_batch["input_ids"]=res["input_ids"]
        data_batch['sft_format'] = text_list
        return data_batch
