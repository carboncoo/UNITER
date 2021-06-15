"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for VE model
"""

from collections import defaultdict
from torch import nn
from torch.nn import functional as F

from .vqa import UniterForVisualQuestionAnswering
from .model import UniterPreTrainedModel, UniterModel, UniterSoftPromptModel, UniterConfig

class UniterForVisualEntailment(UniterForVisualQuestionAnswering):
    """ Finetune UNITER for VE
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim, 3)

class UniterSoftPromptForVisualEntailment(UniterPreTrainedModel):
    """ Finetune UNITER with soft prompts for VQA
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter_softprompt = UniterSoftPromptModel(config, img_dim)
        self.apply(self.init_weights)

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        config = UniterConfig.from_json_file(config_file)
        model = cls(config, *inputs, **kwargs)
        model.uniter_softprompt = UniterSoftPromptModel.from_pretrained(config_file, state_dict, *inputs, **kwargs)
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
        
        predicted_output = sequence_output[:, 19, :] # HARD CODE
        
        # label_mapping = [2160, 2389, 1302] # Yes / Maybe / No
        # label_mapping = [4208, 2654, 1185] # yes / maybe / no
        label_mapping = [1185, 4208, 2654] # no / yes / maybe
        answer_scores = self.uniter_softprompt.cls(predicted_output)[:, label_mapping]
        # import ipdb; ipdb.set_trace()
        
        # pooled_output = self.uniter.pooler(sequence_output)
        # answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores