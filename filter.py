import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

def process_single_sub(sub, folder):
    """处理单个数据项的线程函数"""
    sub["path"] = os.path.join(folder, sub["path"])
    return sub if os.path.exists(sub["path"]) else None

def process_single_file(anno, folder, file_name, output_path, sub_workers=4):
    data_path = os.path.join(anno, file_name)
    output_file = data_path.replace(".pkl", ".json").replace("/work/share1/caption/", output_path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # 读取数据
        data_out = pd.read_pickle(data_path)
        
        # 创建子线程池处理单个文件内的数据
        with ThreadPoolExecutor(max_workers=sub_workers) as executor:
            futures = []
            for sub in data_out:
                futures.append(executor.submit(process_single_sub, sub, folder))
            
            # 使用tqdm显示单个文件处理进度
            filtered_data = []
            with tqdm(total=len(futures), desc=f"Processing {file_name}", leave=False) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        filtered_data.append(result)
                    pbar.update(1)
        
        # 写入结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        return output_file
    
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return None

def filter_data(output_path, file_workers=4, sub_workers=4):
    """双层多线程处理函数"""
    with open("/work/share/projects/mjc/lmfusion/train_files/merge_datac.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建文件处理任务列表
    tasks = []
    for line in data["image_gen"][::-1]:
        anno, folder = line
        files = [f for f in os.listdir(anno) if f.endswith(".pkl")]
        for f in files:
            process_single_file(anno, folder, f, output_path, sub_workers=128)

if __name__ == "__main__":
    #output_json = "/work/share/projects/mjc/data/"
    #filter_data(output_json)
    with open("/work/share/projects/mjc/data/laion-nolang/part3_cap2273274.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))