import json
import cv2
import os
import decord
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_frame(video_path, frame_number ,output_path):
    """
    提取视频的首帧并保存为图片
    
    参数:
        video_path (str): 视频文件路径
        output_path (str): 输出图片路径，如果不指定则保存到视频同目录
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 - {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 - {video_path}")
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    print("frame",frame.shape)
    
    if not ret:
        print("错误：无法读取视频帧")
        return False
    cap.release()
    
    # 设置默认输出路径（视频同目录，文件名加_first_frame.jpg）
    if output_path is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_name}_first_frame.jpg")
    
    # 保存第一帧
    #cv2.imwrite(output_path, frame)
    # print(f"首帧已保存到: {output_path}")
    return frame


def extract_frame_gpu(video_path, frame_number ,output_path):
    if os.path.exists(video_path):
        reader = decord.VideoReader(video_path)
        frame = reader.get_batch([frame_number]).asnumpy()
        frame = np.squeeze(frame, axis=0)
        frame = frame[..., [2, 1, 0]]
        return frame




def get_json_frames(file_path,output_dir,crop=True):
    """
    输出JSON 文件的前十条数据的frame
    参数:
        file_path (str): JSON 文件的路径
    返回:
        None (直接打印前10条数据)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)


        if isinstance(data, (list, tuple)):
  
            print(f"文件 '{file_path}' 的数据：")
            #for i, item in enumerate(data, 1):

            captions_file = os.path.join(output_dir , "captions.txt")

            with open(captions_file, 'w') as f:
                for i, item in enumerate(tqdm(data[:500], desc="处理进度"), 1):
                    #print(f"第 {i} 条数据:")
                    #print(json.dumps(item, ensure_ascii=False, indent=2))
                    print(item['cap'])
                    caption = item['cap'][0] if isinstance(item['cap'], list) else item['cap']

                    frame_number = item['cut'][0]
                    #',frame_number)
                    crop_coords = item['crop']
                    #print("crop",crop_coords)
                    video_path = '/work/'+item ['path']
                    #print("path",video_path)
                    filename = video_path.rsplit('/', 1)[-1].split('.mp4')[0]
                    # print("file",filename)
                    dst_image_name = f"{i:03d}.jpg" 
                    output_path= os.path.join(output_dir,dst_image_name)
                    #print("file",output_path)
                    try:
                        frame=extract_frame_gpu(video_path,frame_number=frame_number,output_path=output_path)
                        if crop:
                            x1, x2, y1, y2 = crop_coords
                            height, width = frame.shape[:2]
                            if x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                                #print(f"错误：无效的裁剪坐标 {crop_coords} (视频尺寸: {width}x{height})")
                                return False
                        
                            frame = frame[y1:y2, x1:x2]

                        cv2.imwrite(output_path, frame)
                        f.write(f"{dst_image_name}: {caption}\n")
                    except Exception as e:
                        #print(f"发生错误: {e}")
                        pass

                #print("-" * 40)
        elif isinstance(data, dict):
            print(f"文件 '{file_path}' 是字典格式，前 10 个键值对：")
            for i, (key, value) in enumerate(list(data.items())[:10], 1):
                print(f"第 {i} 条数据: {key} = {json.dumps(value, ensure_ascii=False, indent=2)}")
                print("-" * 40)
        else:
            print("JSON 数据不是列表或字典，无法打印前 10 条。")
    
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在！")
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式！")
    except Exception as e:
        print(f"发生未知错误: {e}")


def process_one(item):
    frame_number = item['cut'][0]
    crop_coords = item['crop']
    video_path = '/work/'+item ['path']
    filename = video_path.rsplit('/', 1)[-1].split('.mp4')[0]
    output_path= output_dir+filename+f'{frame_number}.jpg'
    if not os.path.exists(output_path):
        if os.path.exists(video_path):
            try:
                reader = decord.VideoReader(video_path)
                frame = reader.get_batch([frame_number]).asnumpy()
                frame = np.squeeze(frame, axis=0)
                frame = frame[..., [2, 1, 0]]
                x1, x2, y1, y2 = crop_coords
                height, width = frame.shape[:2]
                if x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                    print(f"错误：无效的裁剪坐标 {crop_coords} (视频尺寸: {width}x{height})")
                    return False
                frame = frame[y1:y2, x1:x2]
                cv2.imwrite(output_path, frame)
                return True
            except:
                return False
from concurrent.futures import ThreadPoolExecutor, as_completed
def para_process(json_dir,output_dir):
    executor = ThreadPoolExecutor(max_workers=32)
    folder_path = Path(json_dir)  # 替换为你的文件夹路径
    tasks = []
    for json_file in list(folder_path.glob("*.json")):
    #for json_file in folder_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            tasks.append(executor.submit(process_one, item))
    for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
        try:
            state=future.result() 
        except Exception as e:
            print(f"Error processing task: {e}")



def process_dir(json_dir,output_dir):
    folder_path = Path(json_dir)  # 替换为你的文件夹路径

    for json_file in folder_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                print(f"处理文件: {json_file.name}")
                # 你的处理逻辑...
                get_json_frames(json_file,output_dir)
                
            except json.JSONDecodeError:
                print(f"错误: {json_file.name} 不是有效的JSON文件")



# 示例使用
#json_path = "/work/share1/caption/osp/0330/random_video_final_1_3014225.json"  # 替换为你的JSON文件路径
output_dir = "/work/share/projects/mjc/lmfusion/train_files/data_example/video_fames"  # 指定输出目录

#extract_and_save_first_frame(json_path, output_dir)



# 示例使用
#video_path = "/work/share/dataset/xigua_video/videos_clip_chongwu_20241031/subdir_138/6960208519651918366_part5.mp4"

# video_path = "/work/share/dataset/sucai_video/istock_v4/videos_clip_v4_20241111/subdir_272/gm1454703757-490270180_part1.mp4"


# extract_first_frame(video_path=video_path,output_path="/work/share/projects/mjc/lmfusion/train_files/6960208519651918366_part5.jpg")

# video_path = "/work/share/dataset/sucai_video/istock_v4/videos_clip_v4_20241111/subdir_272/gm1454703757-490270180_part1.mp4"
# output_path = "/work/share/projects/mjc/lmfusion/train_files/framesgm1454703757-490270180_part1.jpg"
# extract_first_frame (video_path=video_path,output_path=output_path)


# json_dir='/work/share1/caption/osp/0330'
file_path ='/work/share1/caption/osp/0318/random_video_final_1_1149968.json'
output_dir = "/work/share1/mjc/video_frames/"  # 指定输出目录



# print_first_10_json_items(file_path)

# get_json_frames(file_path,output_dir)

para_process(json_dir,output_dir)
