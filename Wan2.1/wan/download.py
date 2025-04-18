from modules.vae import WanVAE
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


vae = WanVAE(device = "cpu")

save_path = "W/work/share/projects/mjc/lmfusion/Wan2.1/ckpt"
vae.save_pretrained(save_path)