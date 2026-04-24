import torch
import torch.nn as nn

class ImageToTokens(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model,
                 use_class_token=True, use_pos_encoding=True):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.num_patches = (img_size // patch_size) ** 2
        self.use_class_token = use_class_token

        if use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, d_model)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        if self.use_class_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        if self.use_pos_encoding:
            x = x + self.pos_embed

        return x