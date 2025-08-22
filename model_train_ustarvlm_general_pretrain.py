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
        return samples

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        image_name = sample['image']
        conversations = sample['conversations']
        # conv_text = [{"role": "system", "content": 'You are a helpful assistant.'}]
        # for turn in conversations:
        #     if turn['from'] == 'human':
        #         conv_text.append({"role": "user", "content": turn['value']})
        #     elif turn['from'] == 'assistant':
        #         conv_text.append({"role": "assistant", "content": turn['value']})
        ##格局不同llm模型进行调整
        image_pad = '<|image_pad|>' * self.config.image_pad_num
        all_text = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False).replace(
            '<image>', image_pad)
        inputs = self.tokenizer(all_text, return_tensors='pt', max_length=1024,padding=False,truncation=True)
        input_ids = inputs['input_ids'][0].tolist()

        # 构造attention_mask
        attention_mask = [1] * len(input_ids)

        # 构造 labels：只对 assistant 的 response 位置计算 loss，其它位置用 pad_token_id
        labels = [self.tokenizer.pad_token_id] * len(input_ids)
        result_index = self._find_assistant_token(self.tokenizer, input_ids)
        for i in result_index:
            labels[i[0]:i[1]] = input_ids[i[0]:i[1]]

        # 偏移
        input_ids = input_ids[:-1]
        labels = labels[1:]
        attention_mask = attention_mask[:-1]

        image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
        pixel_values = self.processor(text=None, images=image)['pixel_values'][0]


        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

    def _find_assistant_token(self, tokenizer, input_ids):
        result = []
        start_index = 0
        end_index = 0
        while start_index <= len(input_ids) - 1:
            if input_ids[start_index] != tokenizer('assistant')['input_ids'][0]:
                start_index += 1
                end_index += 1
            else:
                end_index += 1
                if input_ids[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                    result.append((start_index + 1, end_index + 1))
                    start_index = end_index + 1
        return result
    
    
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
    "images_path":'/home/dataset0/images/minimind-v_dataset/pretrain_images',
    "data_path": '/home/dataset0/images/minimind-v_dataset/pretrain_vlm_data.jsonl',
    "output_dir": 'save/models/ustarvlm/pretrain_815',
}

def ustarvlm_train():
    config = uStarVLMConfig()
    model = uStarVLM(config)
    print_trainable_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
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
        run_name='ustarvlm_sft_2025_08_21',
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
        ddp_backend="nccl",
        ddp_find_unused_parameters=True,
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
    trainer.save_state()
    
if __name__ == "__main__":
    print("Starting uStarVLM training...")
    ustarvlm_train()
