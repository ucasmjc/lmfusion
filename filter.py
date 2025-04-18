import json
import os
from tqdm import tqdm

def filter_json_by_path(input_file: str, output_file: str) -> None:
    """
    过滤JSON文件中path字段有效的条目并保存到新文件
    
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤有效路径
    filtered_data = []
    for item in tqdm(data, desc="正在过滤数据"):
        path = item.get('path')
        
        # 检查path是否为字符串且路径存在
        if os.path.exists(os.path.join("/work/share/data/imgdata/aes",path)):
            filtered_data.append(item)
    
    # 保存过滤后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    # 打印统计信息
    print(f"处理完成！原始数据：{len(data)}条，过滤后数据：{len(filtered_data)}条")
    print(f"结果已保存至：{os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 配置输入输出路径（根据需要修改）
    input_json = "/work/share1/caption/laion-aes/part0_cap3547693.json"
    output_json = "/work/share/projects/mjc/lmfusion/data/image_gen.json"
    
    # 执行过滤
    filter_json_by_path(input_json, output_json)