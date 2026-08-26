import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models

class SigLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(-10.0))
    
    def forward(self, image_feats, text_feats, pos_labels):
        """
            image_feats: (1, dim)
            text_feats: (B, dim)
            pos_labels: (1, B)
        """
        logits = (image_feats * text_feats).sum(dim=-1, keepdim=True).T
        logits = logits * self.temperature + self.bias
        loss = F.binary_cross_entropy_with_logits(logits, pos_labels, reduction='mean')
        return loss

class ECTCLIP(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.encoder = self._load_encoder()
        self.proj_x4 = nn.Linear(text_dim, 512)
        self.crossattn_x4 = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.i2t_proj = nn.Linear(512, text_dim)
    
    def _load_encoder(self):
        encoder = models.r3d_18(pretrained=True)
        encoder.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        encoder.avgpool = nn.AdaptiveAvgPool3d(1)
        encoder.fc = nn.Identity()
        return encoder

    def forward(self, image, text_feats):
        """
            text_feats: tensor (1, B, text_dim)
            image: tensor (1, image_shape)
        """

        x0 = self.encoder.stem(image)
        x1 = self.encoder.layer1(x0)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        text_feats = text_feats.squeeze(0)  # (B, text_dim)
        query_x4 = self.proj_x4(text_feats)

        x4 = x4.flatten(2).transpose(1, 2).squeeze(0)
        x4, attn_weights = self.crossattn_x4(query_x4, x4, x4)

        image_feats = x4
        image_feats = self.i2t_proj(image_feats)
        text_feats = text_feats

        return image_feats, text_feats


