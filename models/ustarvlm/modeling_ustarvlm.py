
from models.ustarvlm.config import uStarVLMConfig
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

##简单的拼接做法
class uStarVLM(PreTrainedModel):
    # 这里需要注册一下config_class类，不然在用AutoModelForCausalLM加载训练好的VLM时会报错
    config_class = uStarVLMConfig
    def __init__(self, config):
        super(uStarVLM, self).__init__(config)

        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(self.config.llm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_path)
        self.project_layer = nn.Sequential(
            nn.Linear(self.vision_model.config.vision_config.hidden_size * 4, self.llm.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        )
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = True

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):

        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        image_embeddings = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeddings.shape
        # 压缩图片token 这个地方可以优化
        image_embeddings = image_embeddings.view(b, -1, d*4)

        # 对齐image和text
        image_features = self.project_layer(image_embeddings)
        text_embeddings = text_embeddings.to(image_features.dtype)

        # 得到最终输入
        input_embeddings = self.merge_image_features_to_text_embeddings(input_ids, text_embeddings, image_features)

        outputs = self.llm(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fc = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fc(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, **generate_kwargs):
        # 自定义 generate 函数，确保图像信息嵌入后参与整轮 token 的生成。

        if input_ids is None or pixel_values is None:
            raise ValueError("Both input_ids and pixel_values are required for multimodal generation.")

        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        image_embeddings = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeddings.shape
        image_embeddings = image_embeddings.view(b, -1, d * 4)
        image_features = self.project_layer(image_embeddings)
        text_embeddings = text_embeddings.to(image_features.dtype)
        input_embeddings = self.merge_image_features_to_text_embeddings(input_ids, text_embeddings, image_features)

        #构建 attention_mask（如果没给）
        if attention_mask is None:
            attention_mask = torch.ones(input_embeddings.shape[:2], dtype=torch.long, device=input_embeddings.device)

        outputs = self.llm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            **generate_kwargs  # 支持 max_new_tokens, temperature 等常规参数
        )
        return outputs

    def merge_image_features_to_text_embeddings(self, input_ids, text_embeddings, image_features):

        batch, patch, hidde_dim = image_features.shape
        # 找出input_ids中被<image_pad>占位的索引
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        # 将image_features替换原来的<image_pad>的embedding
        text_embeddings[batch_indices, image_indices] = image_features.view(-1, hidde_dim)

        return text_embeddings
