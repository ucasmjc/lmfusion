import pandas as pd
import os
import time,json
import torch
import decord
class DecordDecoder(object):
    def __init__(self, url,ctx=decord.cpu(0), num_threads=70):
        self.num_threads = num_threads
        self.ctx = ctx
        self.reader = decord.VideoReader(url,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)

    def get_avg_fps(self):
        return self.reader.get_avg_fps() if self.reader.get_avg_fps() > 0 else 30.0

    def get_num_frames(self):
        return len(self.reader)

    def get_height(self):
        return self.reader[0].shape[0] if self.get_num_frames() > 0 else 0

    def get_width(self):
        return self.reader[0].shape[1] if self.get_num_frames() > 0 else 0
    def get_frame(self,idx):
        return self.reader.get_batch([idx]).asnumpy()

    # output shape [T, H, W, C]
    def get_batch(self, frame_indices):
        try:
            #frame_indices[0] = 1000
            video_data = self.reader.get_batch(frame_indices).asnumpy()
            video_data = torch.from_numpy(video_data)
            return video_data
        except Exception as e:
            print(f"Get_batch execption: {e}")
            return None
time0=time.time()
aaa=DecordDecoder("/work/share/projects/mjc/lmfusion/7364748394570449445_part37.mp4")
print(aaa.get_frame(258).shape,aaa.get_avg_fps(),aaa.get_num_frames())
time1=time.time()
print(time1-time0)

