import json
import os
import shutil

# 原始 JSON 文件路径

def process_image_data(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 确保数据是列表且至少有 100 条
    if isinstance(data, list) and len(data) >= 100:
        # 创建目标文件夹（如 /work/share/projects/mjc/lmfusion/train_files/data_example/coyo1）
        base_dir = "/work/share/projects/mjc/lmfusion/train_files/data_example"
        folder_name = file_path.split('/512/')[1].split('/')[0]  # 提取 "coyo1"
        target_path = base_dir #os.path.join(base_dir, folder_name)
        os.makedirs(target_path, exist_ok=True)  # 自动创建目录（如果不存在）

        # 保存 caption 到 txt 文件
        captions_file = os.path.join(target_path, "captions.txt")
        with open(captions_file, 'w') as f:
            for idx, item in enumerate(data[:500], 1):
                # 原始图片路径
                src_image_path = os.path.join ("/work/share/data/imgdata/",folder_name,item['path']) 
                # 新图片命名（如 001.jpg, 002.jpg...）
                dst_image_name = f"{idx:03d}.jpg"  # 3位数字，不足补零
                dst_image_path = os.path.join(target_path, dst_image_name)

                # 复制图片（假设 src_image_path 是有效路径）
                if os.path.exists(src_image_path):
                    shutil.copy2(src_image_path, dst_image_path)
                    # 写入 caption（假设 item['cap'] 是字符串列表，取第一个）
                    caption = item['cap'][0] if isinstance(item['cap'], list) else item['cap']
                    f.write(f"{dst_image_name}: {caption}\n")  # 格式如 "001.jpg: a cute cat"
                else:
                    print(f"警告：图片不存在，跳过 {src_image_path}")



        print(f"完成！图片和 captions.txt 已保存到 {target_path}")
    else:
        print("错误：JSON 数据不是列表或不足 100 条")



#file_path = "/work/share1/mjc/imggen_json/512/coyo1/part0_cap1303401.json"
#file_path = "/work/share1/mjc/imggen_json/512/coyo2/part0_cap1640462.json"
#file_path = "/work/share1/mjc/imggen_json/512/laion-aes/part0_cap461103.json"
#file_path = "/work/share1/mjc/imggen_json/512/laion-multi/part0_cap622177.json"
#file_path = "/work/share1/mjc/imggen_json/512/laion-nolang/part0_cap358902.json"
file_path = "/work/share1/mjc/imggen_json/512/recap1/part0_cap903071.json"
#file_path = "/work/share1/mjc/imggen_json/512/recap2/part0_cap292846.json"



process_image_data(file_path)
