# extract_qwen25_vit.py
import os, json, torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_ID = "/home/dataset1/gaojing/models/Qwen2.5-VL-3B-Instruct"   # 也可换成 3B/32B/72B
SAVE_DIR = "/home/dataset0/images/qwen2.5-vit"


def main():
    # 建议使用较新的 transformers 版本（>=4.51）；官方文档/博客在 2025 年初合入该模型。 
    # 可按需设置 dtype / device_map
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    vision = model.visual  # 视觉编码器（ViT），Qwen2.5-VL里模块名为 visual

    # 1) 保存视觉 state_dict（safetensors 更安全，这里用 torch.save 也行）
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(vision.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))

    # 2) 保存视觉 config（从复合 config 中取 vision_config）
    vision_cfg = model.config.vision_config.to_dict()
    # 记录来源模型，方便追溯
    meta = {
        "source_model": MODEL_ID,
        "note": "Qwen2.5-VL visual encoder (ViT) extracted from model.visual",
    }
    with open(os.path.join(SAVE_DIR, "vision_config.json"), "w", encoding="utf-8") as f:
        json.dump(vision_cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(SAVE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Vision encoder saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
