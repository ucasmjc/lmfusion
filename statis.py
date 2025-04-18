
import pandas as pd
import os
def get_datasamples(data_base_path, return_type="list",suffix=".json"):
    if os.path.isdir(data_base_path):
        results=[]
        files=os.listdir(data_base_path)
        for f in files:
            if f.endswith(suffix):
                try:
                    data_path=os.path.join(data_base_path,f)
                    if data_path.endswith(".json"):
                        data_out = pd.read_json(data_path)
                    elif data_path.endswith(".pkl"):
                        data_out = pd.read_pickle(data_path)
                    elif data_path.endswith(".jsonl"):
                        data_out = pd.read_json(data_path, lines=True)
                    elif data_path.endswith(".parquat"):
                        data_out = pd.read_parquat(data_path)
                    else:
                        raise NotImplementedError(f"Unsupported file format: {data_path}")
                    if isinstance(data_out, pd.DataFrame):
                        data_out=data_out.to_dict("records")
                    results+=data_out
                    print("load",data_path)
                except:
                    continue
        return results
    else:
        data_path=data_base_path
        if data_path.endswith(".json"):
            data_out = pd.read_json(data_path)
        elif data_path.endswith(".pkl"):
            data_out = pd.read_pickle(data_path)
        elif data_path.endswith(".jsonl"):
            data_out = pd.read_json(data_path, lines=True)
        elif data_path.endswith(".parquat"):
            data_out = pd.read_parquat(data_path)
        else:
            raise NotImplementedError(f"Unsupported file format: {data_path}")
    if return_type == "list":
        if isinstance(data_out, pd.DataFrame):
            return data_out.to_dict("records")
        elif isinstance(data_out, list):
            return data_out
    else:
        raise NotImplementedError(f"Unsupported return_type: {return_type}")
import time
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
def inter(path):
    try:
        Image.open(path)
        return True
    except:
        return False
import pickle
def load_dataset(data_path):
    cap_lists = []
    executor = ThreadPoolExecutor(max_workers=120)
    with open(data_path, "r") as f:
        folder_anno = [
            i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0
        ]
    output_dir="/work/share1/mjc/img_data"
    for folder, anno in folder_anno:
        sub_list = get_datasamples(anno,suffix=".pkl")
        existing_items = []  # 当前anno的有效项目
        for sub in tqdm(sub_list, desc=f"Checking {anno}", leave=False):
            if 'image_gen' in sub:
                sub["path"]=sub['image_gen']
            sub["path"] = os.path.join(folder, sub["path"])
        cap_lists += sub_list
    return cap_lists

def get_cap_list(data_path):
    cap_lists = []
    executor = ThreadPoolExecutor(max_workers=120)
    with open(data_path, "r") as f:
        folder_anno = [
            i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0
        ]
    output_dir="/work/share1/mjc/img_data"
    for folder, anno in folder_anno:
        sub_list = get_datasamples(anno,suffix=".pkl")
        print(f"\n{anno}: Original items: {len(sub_list)}")
def filter_data(data_path):
    cap_lists = []
    with open(data_path, "r") as f:
        folder_anno = [
            i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0
        ]
    for folder, anno in folder_anno:
        sub_list = get_datasamples(anno,suffix=".pkl")
        print(f"Building {anno}...")
        for sub in sub_list:
            sub["path"] = os.path.join(folder, sub["path"])
        print(len(sub_list))
        cap_lists += sub_list
    return cap_lists

get_cap_list("/work/share/projects/mjc/lmfusion/data/image_gen.txt")
#import time
#time0=time.time()
#aaa=load_dataset("/work/share/projects/mjc/lmfusion/data/image_gen.txt")
#time1=time.time()
#print(len(aaa),time1-time0)