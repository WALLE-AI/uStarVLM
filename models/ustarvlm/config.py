from transformers import PretrainedConfig

class uStarVLMConfig(PretrainedConfig):
    model_type = 'vlm_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_path = r'/home/dataset0/images/Qwen3-0.6B'
        self.vision_model_path = r'/home/dataset1/gaojing/models/siglip2-base-patch16-224'
        self.image_pad_num = 49
        self.freeze_vision_model = True  # 是否冻结视觉模型进行训练
        self.freeze_llm = False # 是否冻结语言模型
        self.num_layers_to_unfreeze = 2 # 开放llm的前几层参与训练