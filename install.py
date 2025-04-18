import torch
import torch.nn as nn
# 生成合法掩码（假设每个head选前256个位置）
temp_q_img_mask = torch.zeros((2, 12, 401), dtype=torch.bool)
temp_q_img_mask[:, :, :256] = True  # 形状 [2,12,401]，含6144个True
v_query_states = torch.ones((2, 12, 401,16))
print(v_query_states[temp_q_img_mask].shape)
sys.exit()
# 调整右侧张量形状
v_query_flat = v_query_states.contiguous().view(-1, 16)  # 形状 [6144, 128]

# 强制校验选中数量
num_selected = temp_q_img_mask.sum().cpu().item()
assert num_selected == 6144, f"掩码True数量应为6144，当前为{num_selected}"

# 执行严格的形状对齐操作
try:
    torch.npu.set_check_enable(True)
    t_query_states[temp_q_img_mask] = v_query_flat
except RuntimeError as e:
    print(f"赋值失败，后端错误: {str(e)}")
    raise
finally:
    torch.npu.set_check_enable(False)
