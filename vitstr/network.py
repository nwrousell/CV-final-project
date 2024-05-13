import torch
from torch import nn
from transformers import AutoImageProcessor, DeiTModel

from .config import embedding_dim, characters

class ViTSTR(nn.Module):
    def __init__(self):
        super(ViTSTR, self).__init__()

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
        self.DeiT = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
        self.head = nn.Linear(in_features=embedding_dim, out_features=len(characters)+2)

    def forward(self, x, max_seq_length, device):
        # process image
        inputs = self.image_processor(x, return_tensors="pt")['pixel_values'].to(device)

        # DeiT to get encodings
        outputs = self.DeiT(inputs)
        encodings = outputs.last_hidden_state # (batch_size, 198, 192)
        # 198 = 196 patch encodings, 1 class token, and 1 distill token
        
        # apply head proj
        preds = self.head(encodings[:, :max_seq_length, :])
        return preds