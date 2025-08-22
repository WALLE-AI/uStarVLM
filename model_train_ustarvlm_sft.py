import os
import shutil
from models.ustarvlm.config import uStarVLMConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
import torch
from PIL import Image
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any

from models.ustarvlm.modeling_ustarvlm import uStarVLM
from utils import print_trainable_parameters
from transformers import AutoConfig, AutoModelForCausalLM
from models.ustarvlm.config import uStarVLMConfig
from models.ustarvlm.modeling_ustarvlm import uStarVLM

class uStarVLMDataset(Dataset):
    def __init__(self, iamge_path, text_json_path, tokenizer, processor, config):
        super(uStarVLMDataset, self).__init__()
        self.images_path = iamge_path
        self.text_json_path = text_json_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # with open(text_json_path, 'r', encoding='utf-8') as f:
        #     self.datas = json.load(f)
        self.datas = self.load_data(text_json_path)
            
            
    def load_data(self,path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
                if line_num == 20000:
                    break
        return samples

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        image_name = sample['image']
        conversations = sample['conversations']

        # 1) 用 chat_template 生成纯文本模板
        raw_text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )

        # 2) 用空格分隔铺设 image_pad（更稳健）
        image_pad = " ".join(["<|image_pad|>"] * self.config.image_pad_num)
        raw_text = raw_text.replace("<image>", image_pad)

        # 3) tokenize
        enc = self.tokenizer(
            raw_text, return_tensors='pt', max_length=1024, padding=False, truncation=True
        )
        input_ids = enc['input_ids'][0].tolist()
        attention_mask = [1] * len(input_ids)

        # 4) 仅对 assistant 段打标签：匹配 <|im_start|>assistant\n ... <|im_end|>
        labels = [self.tokenizer.pad_token_id] * len(input_ids)
        for st, ed in self._find_assistant_spans(self.tokenizer, input_ids):
            labels[st:ed] = input_ids[st:ed]

        # 5) 右移对齐（手动 shift-one）
        input_ids = input_ids[:-1]
        labels = labels[1:]
        attention_mask = attention_mask[:-1]

        # 6) 图像
        image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
        pixel_values = self.processor(text=None, images=image)['pixel_values'][0]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

    def _find_assistant_spans(self, tokenizer, input_ids):
        """鲁棒地定位 <|im_start|>assistant\n  到  <|im_end|> 的内容区间。"""
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")
        marker   = tokenizer.encode("assistant\n", add_special_tokens=False)

        spans, i, n = [], 0, len(input_ids)
        m = len(marker)
        while i < n:
            if input_ids[i] == im_start and i + 1 + m <= n and input_ids[i+1:i+1+m] == marker:
                j = i + 1 + m
                while j < n and input_ids[j] != im_end:
                    j += 1
                # 内容区间是 [start=j_begin, end=j)
                spans.append((i + 1 + m, j))
                i = j + 1
            else:
                i += 1
        return spans

    # def __getitem__(self, index):
    #     sample = self.datas[index]
    #     image_name = sample['image']
    #     conversations = sample['conversations']
    #     # conv_text = [{"role": "system", "content": 'You are a helpful assistant.'}]
    #     # for turn in conversations:
    #     #     if turn['from'] == 'human':
    #     #         conv_text.append({"role": "user", "content": turn['value']})
    #     #     elif turn['from'] == 'assistant':
    #     #         conv_text.append({"role": "assistant", "content": turn['value']})
    #     ##格局不同llm模型进行调整
    #     image_pad = '<|image_pad|>' * self.config.image_pad_num
    #     all_text = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False).replace(
    #         '<image>', image_pad)
    #     inputs = self.tokenizer(all_text, return_tensors='pt', max_length=1024,padding=False,truncation=True)
    #     input_ids = inputs['input_ids'][0].tolist()

    #     # 构造attention_mask
    #     attention_mask = [1] * len(input_ids)

    #     # 构造 labels：只对 assistant 的 response 位置计算 loss，其它位置用 pad_token_id
    #     labels = [self.tokenizer.pad_token_id] * len(input_ids)
    #     result_index = self._find_assistant_token(self.tokenizer, input_ids)
    #     for i in result_index:
    #         labels[i[0]:i[1]] = input_ids[i[0]:i[1]]

    #     # 偏移
    #     input_ids = input_ids[:-1]
    #     labels = labels[1:]
    #     attention_mask = attention_mask[:-1]

    #     image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
    #     pixel_values = self.processor(text=None, images=image)['pixel_values'][0]


    #     return {
    #         'input_ids': input_ids,
    #         'labels': labels,
    #         'attention_mask': attention_mask,
    #         'pixel_values': pixel_values
    #     }

    # def _find_assistant_token(self, tokenizer, input_ids):
    #     result = []
    #     start_index = 0
    #     end_index = 0
    #     while start_index <= len(input_ids) - 1:
    #         if input_ids[start_index] != tokenizer('assistant')['input_ids'][0]:
    #             start_index += 1
    #             end_index += 1
    #         else:
    #             end_index += 1
    #             if input_ids[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
    #                 result.append((start_index + 1, end_index + 1))
    #                 start_index = end_index + 1
    #     return result
 
    
class uStarVLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f['input_ids']) for f in features)
        input_ids, labels, attention_mask, pixel_values = [], [], [], []
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(f['labels'] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f['attention_mask'] + [0] * pad_len)
            pixel_values.append(f['pixel_values'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'pixel_values': torch.stack(pixel_values)
        }


data_args = {
    "images_path":'/home/dataset0/images/minimind-v_dataset/sft_images',
    "data_path": '/home/dataset0/images/minimind-v_dataset/sft_vlm_data.jsonl',
    "output_dir": 'save/models/ustarvlm/domain_sft_0819',
}



# def ustarvlm_continue_train(prev_dir,config):
    
#     AutoConfig.register("vlm_model", uStarVLMConfig)
#     AutoModelForCausalLM.register(uStarVLMConfig, uStarVLM)
#     # 2. 从 checkpoint 载入 config 和权重
#     # 可在 from_pretrained 时覆写一些开关（示例）：
#     config.freeze_llm = False                 # 领域预训练通常需要微调 LLM
#     config.freeze_vision_model = True         # 通常冻结视觉 backbone
#     config.num_layers_to_unfreeze = 4         # 若只想放开部分层，下面会按该值解冻

#     model = uStarVLM.from_pretrained(prev_dir, config=config).cuda()

#     tokenizer = AutoTokenizer.from_pretrained(prev_dir if os.path.exists(os.path.join(prev_dir, "tokenizer_config.json")) 
#                                               else config.llm_path, use_fast=True)
#     processor = AutoProcessor.from_pretrained(prev_dir if os.path.exists(os.path.join(prev_dir, "preprocessor_config.json")) 
#                                               else config.vision_model_path)
#     return model, tokenizer, processor

def ustarvlm_continue_train(prev_dir, config):
    AutoConfig.register("vlm_model", uStarVLMConfig)
    AutoModelForCausalLM.register(uStarVLMConfig, uStarVLM)

    # 载入模型
    model = uStarVLM.from_pretrained(prev_dir, config=config)

    # 载入 tokenizer / processor（优先 ckpt；否则回退到原始路径）
    tokenizer = AutoTokenizer.from_pretrained(
        prev_dir if os.path.exists(os.path.join(prev_dir, "tokenizer_config.json"))
        else config.llm_path, use_fast=True
    )
    processor = AutoProcessor.from_pretrained(
        prev_dir if os.path.exists(os.path.join(prev_dir, "preprocessor_config.json"))
        else config.vision_model_path
    )

    # === 关键：把 <|image_pad|>, <image> 注册成“单一特殊 token” ===
    specials = []
    for tok in ["<|image_pad|>", "<image>"]:
        if tok not in tokenizer.get_vocab():
            specials.append(tok)
    if specials:
        tokenizer.add_special_tokens({"additional_special_tokens": specials})
        # 扩展 LLM embedding 尺寸
        model.llm.resize_token_embeddings(len(tokenizer))

    # 建议：把 tokenizer/processor 存到输出目录，方便推理端直接读取
    # （训练流程结束也会再次保存）
    return model, tokenizer, processor

def ustarvlm_train():
    prev_dir = "/home/dataset1/gaojing/llm/uStarVLM/save/models/ustarvlm/pretrain_815"
    config = uStarVLMConfig.from_pretrained(prev_dir)
    model, tokenizer, processor = ustarvlm_continue_train(prev_dir,config)
    print_trainable_parameters(model)

    if os.path.exists(data_args["output_dir"]):
        shutil.rmtree(data_args["output_dir"])
    os.makedirs(data_args["output_dir"], exist_ok=True)
    
    train_datasets = uStarVLMDataset(data_args["images_path"], data_args['data_path'], tokenizer, processor, config)

    args = TrainingArguments(
        output_dir=data_args["output_dir"],
        do_train=True,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_accumulation_steps=1,
        logging_steps=10,
        report_to='swanlab',
        run_name='domain_sft_0821',
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_datasets,
        data_collator=uStarVLMDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(data_args["output_dir"])
    # tokenizer.save_pretrained(data_args["output_dir"])
    # processor.save_pretrained(data_args["output_dir"])
    trainer.save_state()
    
if __name__ == "__main__":
    print("Starting uStarVLM training...")
    ustarvlm_train()
