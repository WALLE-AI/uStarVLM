import os
import json
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset

class uStarVLMDataset(Dataset):
    def __init__(self, image_dir, jsonl_path, tokenizer, processor, config):
        super().__init__()
        self.image_dir = image_dir
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.samples = self._load(jsonl_path)

    def _load(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def _find_assistant_token_spans(self, tokenizer, input_ids):
        # 依据 chat_template：查找 "<|im_start|>assistant" 到 "<|im_end|>" 之间的片段
        # 对于 Qwen 系列，这种做法通常可行；如需更稳健，可直接用模板时返回 role mask。
        result = []
        start = 0
        im_end = tokenizer("<|im_end|>")["input_ids"][0]
        assistant_prefix_ids = tokenizer("<|im_start|>assistant")["input_ids"]
        L = len(input_ids)
        i = 0
        while i <= L - len(assistant_prefix_ids):
            if input_ids[i : i + len(assistant_prefix_ids)] == assistant_prefix_ids:
                # 内容从 prefix 之后开始计 loss
                j = i + len(assistant_prefix_ids)
                # 找最近的 <|im_end|>
                while j < L and input_ids[j] != im_end:
                    j += 1
                if j < L:
                    # (i+len(prefix), j+1) 覆盖 assistant 的内容（含 im_end）
                    result.append((i + len(assistant_prefix_ids), j + 1))
                    i = j + 1
                else:
                    break
            else:
                i += 1
        return result

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        conversations = sample["conversations"]

        # 把 <image> 替换为若干 <|image_pad|>
        image_pad = "<|image_pad|>" * self.config.image_pad_num
        text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        ).replace("<image>", image_pad)

        enc = self.tokenizer(text, return_tensors="pt", max_length=1024, padding=False, truncation=True)
        input_ids = enc["input_ids"][0].tolist()
        attention_mask = [1] * len(input_ids)

        # 只计算 assistant 文本的 loss
        labels = [self.tokenizer.pad_token_id] * len(input_ids)
        for s, e in self._find_assistant_token_spans(self.tokenizer, input_ids):
            labels[s:e] = input_ids[s:e]

        # 右移一位（和 causal LM 对齐）
        input_ids = input_ids[:-1]
        labels = labels[1:]
        attention_mask = attention_mask[:-1]

        # 读图并处理
        # image = Image.open(image_path).convert("RGB")
        # pixel_values = self.processor(text=None, images=image, return_tensors="pt")["pixel_values"][0]
        image = Image.open(os.path.join(image_path)).convert("RGB")
        pixel_values = self.processor(text=None, images=image)['pixel_values'][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }


class uStarVLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask, pixel_values = [], [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(f["labels"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            # 直接用 bf16，降低显存
            pixel_values.append(f["pixel_values"].to(torch.bfloat16))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values),
        }
