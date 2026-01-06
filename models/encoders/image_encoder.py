import torch.nn as nn
from transformers import AutoModel

    

class DINOv2Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze = True
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    

    def forward(self, images):

        outputs = self.model(images)
        image_tokens = outputs.last_hidden_state          # b, 257(16*16+1), 768
        return image_tokens
    