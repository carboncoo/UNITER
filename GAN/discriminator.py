import os
import sys
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

sys.path.append(os.path.join(os.path.abspath(__file__+'/../..'))) # UNITER/
from model.model import UniterModel, UniterPreTrainedModel
from model.layer import GELU

class UniterModelforSequenceClassification(UniterPreTrainedModel):
    """ Finetune UNITER for authentic/pseudo data classification
    """
    def __init__(self, config, img_dim, num_labels=2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_labels)
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
        logits = self.classifier(pooled_output)
        
        if compute_loss:
            targets = batch['targets'].argmax(-1)
            loss = F.cross_entropy(
                logits, targets, reduction='none')
            return loss
        else:
            return logits
        
        
        

if __name__ == '__main__':
    pass