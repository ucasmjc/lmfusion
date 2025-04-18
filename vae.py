import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    "/work/share/projects/mjc/lmfusion/vae", 
    torch_dtype=torch.float16        
).to(device="npu")

# Load an image
image_path = "/work/share/projects/mjc/lmfusion/nkp.jpg"  # Replace with your image path
pil_image = Image.open(image_path).convert("RGB")
# 预处理图像
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image_tensor = transform(pil_image).unsqueeze(0).to(device="npu",dtype=torch.float16)

with torch.no_grad():
    z= vae.encode(image_tensor).latent_dist.sample()
    print(z.shape)
    res=vae.decode(z).sample

# 定义函数将张量转换为PIL图像
def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu().float()  # 移除批次维度并转为浮点型
    tensor = (tensor + 1) / 2  # 反归一化到[0, 1]
    tensor = tensor.clamp(0, 1)  # 限制范围
    return transforms.ToPILImage()(tensor)

# 转换为PIL图像
reconstructed_pil = tensor_to_pil(res)

# 保存重建图像
reconstructed_pil.save("reconstructed_image11.jpg")



# 可视化（可选）
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(pil_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("重建图像")
plt.imshow(reconstructed_pil)
plt.axis('off')
plt.savefig("scale_res.png")