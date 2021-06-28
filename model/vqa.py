"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel, UniterSoftPromptModel


class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores
        
        
class UniterSoftPromptForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER with soft prompts for VQA
    """
    def __init__(self, config, img_dim, *inputs, **kwargs):
        super().__init__(config)
        self.uniter_softprompt = UniterSoftPromptModel(config, img_dim, *inputs, **kwargs)
        self.apply(self.init_weights)
        self.uniter_softprompt.set_hard_prompt('[MASK] It is just .')
        self.label_mapping = kwargs.get('label_mapping', [0])

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        config = UniterConfig.from_json_file(config_file)
        model = cls(config, *inputs, **kwargs)
        model.uniter_softprompt = UniterSoftPromptModel.from_pretrained(config_file, state_dict, *inputs, **kwargs)
        model.uniter_softprompt.set_hard_prompt('[MASK] It is just .')
        
        # TODO: 
        #   use vqa label_mapping to reset "model.uniter_softprompt.cls.predictions.decoder"
        #   for class described with multiple words, use their mean pooling
        #   make sure to also set bias accordingly
        
        class_weights = model.uniter_softprompt.uniter.embeddings.word_embeddings.weight[kwargs.get('label_mapping', [0])].clone()
        model.uniter_softprompt.cls.predictions.decoder = nn.Linear(
                                    class_weights.size(1),
                                    class_weights.size(0),
                                    bias=False)
        model.uniter_softprompt.cls.predictions.decoder.weight.data = class_weights
        model.uniter_softprompt.cls.predictions.bias = nn.Parameter(model.uniter_softprompt.cls.predictions.bias.data[kwargs.get('label_mapping', [0])].clone())
        return model

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter_softprompt(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        
        predicted_output = sequence_output[:, self.uniter_softprompt.prediction_pos, :]
        answer_scores = self.uniter_softprompt.cls(predicted_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores