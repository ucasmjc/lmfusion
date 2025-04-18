import json
import os
from tqdm import tqdm
def transform_image_paths(json_file):
    # 读取原始JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    datasets=data["image_gen"]
    count=0
    alljson=[]
    for dataset in datasets:
        path, cap = dataset
        if os.path.isdir(path):
            alljson.extend(
                [[os.path.join(path, f), cap] 
                    for f in os.listdir(path) if f.endswith('.json')]
            )
        else:
            alljson.append(dataset)
    for dataset in alljson:
        path, cap = dataset
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        new_json=[]
        count=0
        for item in tqdm(data):
            if os.path.exists(os.path.join(cap, item['path'])):
                new_json.append(item)
                count+=1
        save_path=path.replace(".json",f"_new{count}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_json,f)
        print(count,len(data))

if __name__ == "__main__":
    json_file = "/work/share/projects/mjc/lmfusion/train_files/merge_data.json"
    transform_image_paths(json_file)