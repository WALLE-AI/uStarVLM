# run_vit_feature.py
import json, torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,  # 用来构造同结构的 visual
)

EXTRACTED_DIR = "/home/dataset0/images/qwen2.5-vit"
MODEL_ID = "/home/dataset1/gaojing/models/Qwen2.5-VL-3B-Instruct"  # 需与抽取时一致

class Qwen25ViT(nn.Module):
    """
    轻包装：底层仍复用官方实现中的 visual 结构，保证权重完全对齐。
    """
    def __init__(self, reference_model_id, extracted_dir, dtype=torch.float16, device="cuda"):
        super().__init__()
        # 构建一个临时模型，只为拿到结构一致的 visual 模块
        tmp = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reference_model_id, torch_dtype=dtype, device_map={"": device}
        )
        self.visual = tmp.visual  # 结构
        # 加载我们保存的 state_dict
        sd = torch.load(f"{extracted_dir}/pytorch_model.bin", map_location=device)
        self.visual.load_state_dict(sd, strict=True)
        self.visual.to(device=device, dtype=dtype)
        self.device = device
        # 可选：删掉其余部分释放显存
        del tmp

    @torch.no_grad()
    def forward(self, pixel_values, image_grid_thw):
        # Qwen2.5-VL ViT 前向一般需要这两个键（动态分辨率所需）
        out = self.visual(hidden_states=pixel_values, grid_thw=image_grid_thw)
        # 通常包含 last_hidden_state（形状 ~ [B, N_tokens, C] 或扁平变体）
        return out

# def preprocess_images(paths, model_id, min_pixels=None, max_pixels=None, device="cuda"):
#     processor = AutoProcessor.from_pretrained(
#         model_id,
#         min_pixels=min_pixels,   # 可控制视觉 token 预算（动态分辨率）
#         max_pixels=max_pixels,
#     )
#     imgs = [Image.open(p).convert("RGB") for p in paths]
#     # 直接用处理器拿到 pixel_values 与 image_grid_thw
#     batch = processor(images=imgs, return_tensors="pt")
#     # 有些环境下键名在 batch.image_grid_thw / batch["image_grid_thw"]
#     pixel_values = batch["pixel_values"].to(device)
#     image_grid_thw = batch["image_grid_thw"].to(device)
#     return pixel_values, image_grid_thw

def preprocess_images(paths, model_id, min_pixels=None, max_pixels=None, device="cuda"):
    imgs = [Image.open(p).convert("RGB") for p in paths]

    # 只取图像处理器，避免 processor.__call__ 对 text 的依赖
    proc = AutoProcessor.from_pretrained(model_id)
    ip = proc.image_processor

    # 按需设置动态分辨率预算（有些版本可直接传参，有些需要覆盖属性）
    if min_pixels is not None: ip.min_pixels = min_pixels
    if max_pixels is not None: ip.max_pixels = max_pixels

    batch = ip(images=imgs, return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device)
    image_grid_thw = batch["image_grid_thw"].to(device)  # 关键：动态分辨率所需

    return pixel_values, image_grid_thw

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit = Qwen25ViT(MODEL_ID, EXTRACTED_DIR, dtype=torch.bfloat16, device=device)
    pixel_values, image_grid_thw = preprocess_images(
        ["./resource/images/test/dog.png"], MODEL_ID,
        # 例如让每张图大致 256~1024 个视觉 token
        min_pixels=256*28*28, max_pixels=1024*28*28,
        device=device
    )
    with torch.no_grad():
        out = vit(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    feats = out.last_hidden_state  # [B, N, C]（具体形状因图像大小而变）
    pooled = feats.mean(dim=1)     # 简单平均池化得到单向量特征
    print("features:", feats.shape, "pooled:", pooled.shape)
