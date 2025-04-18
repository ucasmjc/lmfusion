import csv
import json
import os
from io import StringIO

def load_input_file(input_path):
    """ 加载输入文件并建立行号索引 """
    line_map = []
    with open(input_path, 'r') as jf:
        datasets = json.load(jf)
    for line_num,dataset in enumerate(datasets["image_under"]):
        json_path = dataset[0]
        image_root = dataset[1]
        line_map.append( (json_path, image_root) )
                
    return line_map
from tqdm import tqdm
def fix_image_field(original_path):
    """ 修复单个文件的image字段 """
    try:
        with open(original_path, 'r', encoding='utf-8') as f:
            datafile = json.load(f)
    except Exception as e:
        return None, f"文件读取失败: {str(e)}"
    
    # 修复逻辑
    for data in tqdm(datafile):
        original_image = data.get('image')
        if not isinstance(original_image, list):
            if isinstance(original_image, str):
                data['image'] = [original_image]
            elif original_image is None:
                data['image'] = []
            else:  # 处理数字、字典等其他类型
                data['image'] = [str(original_image)]
            modified = True
    
    # 生成新文件名
    base_dir = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    new_name = f"{os.path.splitext(base_name)[0]}_fixed.json"
    new_path = original_path
    
    # 保存文件
    try:
        with open(new_path, 'w', encoding='utf-8',errors='ignore') as f:
            json.dump(datafile, f, indent=2, ensure_ascii=False)
        return new_path, None
    except Exception as e:
        return None, f"文件保存失败: {str(e)}"

def main(input_file):
    line_map = load_input_file(input_file)
    total_lines = len(line_map)
    
    print(f"\n成功加载 {total_lines} 行数据，其中有效行 {len([x for x in line_map if x])} 行")
    print("输入格式说明：")
    print("  - 单行号: 5")
    print("  - 多行号: 3,5,7")
    print("  - 范围: 8-12")
    print("  - 退出: exit\n")
    
    while True:
        user_input = input("请输入要处理的行号 (或输入exit退出): ").strip()
        if user_input.lower() == 'exit':
            break
            
        try:
            # 解析行号输入
            selected = set()
            for part in user_input.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected.update(range(start, end+1))
                else:
                    selected.add(int(part))
            
            # 验证行号有效性
            valid_lines = []
            for line_num in sorted(selected):
                if line_num < 1 or line_num > total_lines:
                    print(f"忽略无效行号: {line_num} (超出范围)")
                    continue
                if line_map[line_num-1] is None:
                    print(f"忽略无效行号: {line_num} (原始数据解析失败)")
                    continue
                valid_lines.append(line_num)
            
            if not valid_lines:
                print("没有有效的可处理行号")
                continue
                
            # 批量处理
            for line_num in valid_lines:
                json_path, image_root = line_map[line_num-1]
                print(f"\n▶ 正在处理第 {line_num} 行")
                print(f"   JSON路径: {json_path}")
                
                if not os.path.exists(json_path):
                    print("❌ 错误: JSON文件不存在")
                    continue
                
                new_path, error = fix_image_field(json_path)
                if error:
                    print(f"❌ 修复失败: {error}")
                else:
                    print(f"✅ 已生成修复文件: {new_path}")
                
        except ValueError:
            print("输入格式错误，请重新输入")
        except Exception as e:
            print(f"处理异常: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python fixer.py 输入文件.txt")
        sys.exit(1)
    
    if not os.path.exists(sys.argv[1]):
        print(f"错误: 输入文件 {sys.argv[1]} 不存在")
        sys.exit(2)
    
    main(sys.argv[1])