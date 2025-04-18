import json
import os

def transform_image_paths(json_file):
    # 读取原始JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个项目的路径
    for item in data:
        if isinstance(item.get('image'), list):
            # 转换路径并去重前导斜杠
            item['image'] = [path.lstrip('/') for path in item['image']]
    
    # 覆盖写入原始文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"成功处理 {len(data)} 条数据，文件已更新于: {os.path.abspath(json_file)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python transform_paths.py your_file.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"错误: 文件 {json_file} 不存在")
        sys.exit(2)
    
    try:
        transform_image_paths(json_file)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(3)