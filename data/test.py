from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="Lin-Chen/ShareGPT4V",
    filename="sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
    repo_type="dataset",
    local_dir="mjc",  # 保存到本地指定路径
    local_dir_use_symlinks=False  # 避免符号链接，直接复制文件
)