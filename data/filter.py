import csv
import json
import os
from io import StringIO

def parse_line_with_csv(line):
    """ 使用CSV解析器处理带引号的字段 """
    try:
        # 创建CSV读取器
        reader = csv.reader(StringIO(line.strip()),
                          delimiter=',',
                          quotechar='"',
                          skipinitialspace=True)
        # 获取第一个有效行
        row = next(reader)
        if len(row) != 2:
            raise ValueError("必须包含且仅包含两个字段")
        return [field.strip() for field in row]
    except Exception as e:
        raise ValueError(f"CSV解析失败: {str(e)}")

def validate_line_format(line):
    """ 新版行解析验证 """
    try:
        # 使用CSV解析器处理
        json_rel, image_root = parse_line_with_csv(line)
        
        # 基础验证
        if not json_rel.endswith('.json'):
            raise ValueError("JSON路径必须以.json结尾")
        if not all(os.path.isabs(p) or os.path.exists(p) for p in [json_rel, image_root]):
            print("[警告] 检测到相对路径，建议使用绝对路径")
        
        return json_rel, image_root
    except Exception as e:
        raise ValueError(f"行格式验证失败: {str(e)}")

def check_json_structure(data):
    errors = []
    # 检查image字段
    if 'image' not in data:
        errors.append("Missing 'image' key")
    else:
        image_list = data['image']
        if not isinstance(image_list, list):
            errors.append("'image' is not a list")
        elif len(image_list) < 1:
            errors.append("'image' list is empty")
        else:
            if not isinstance(image_list[0], str):
                errors.append("First item in 'image' is not a string")
    
    # 检查conversations字段
    if 'conversations' not in data:
        errors.append("Missing 'conversations' key")
    else:
        convs = data['conversations']
        item=convs[0]
        if not isinstance(convs, list):
            errors.append("'conversations' is not a list")
        elif len(convs) < 2:
            errors.append("'conversations' has fewer than 2 items")
        else:
            if 'from' not in item or item['from'] != 'human':
                errors.append(f"Item  in 'conversations' is not from 'human'")
            if 'value' not in item:
                errors.append(f"Item in 'conversations' is missing 'value' key")
    return errors

def main(input_file):
    with open(input_file, 'r') as jf:
        datasets = json.load(jf)
    for line_num,dataset in enumerate(datasets["image_under"]):
        json_path = dataset[0]
        image_root = dataset[1]
        if not os.path.exists(json_path):
            print(f"Line {line_num}: JSON file not found at '{json_path}'\n   Content: {json_path}")
            continue
        errors = []
        data = None
        # 解析JSON文件
        try:
            with open(json_path, 'r') as jf:
                data = json.load(jf)
        except Exception as e:
            errors.append(f"JSON parsing error: {e}")

        # 检查JSON结构
        try:
            data = data[0]
        except Exception as e:
            print(data,json_path)
        if data is not None:
            structure_errors = check_json_structure(data)
            errors.extend(structure_errors)

            # 检查图片路径是否存在
            if 'image' in data and isinstance(data['image'], list) and len(data['image']) > 0 and isinstance(data['image'][0], str):
                image_rel = data['image'][0]
                full_image_path = os.path.join(image_root, image_rel)
                if not os.path.exists(full_image_path):
                    errors.append(f"Image file not found: '{full_image_path}'")
                elif not os.path.isfile(full_image_path):
                    errors.append(f"Path is not a file: '{full_image_path}'")

        # 输出错误
        if errors:
            print(f"Line {line_num} has errors:")
            for error in errors:
                print(f"  - {error}")
            print(data)

if __name__ == "__main__":
    main("/work/share/projects/mjc/lmfusion/train_files/merge_data.json")