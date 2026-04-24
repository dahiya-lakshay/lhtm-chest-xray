import torch.nn as nn
from model.image_embeddings import ImageToTokens

class BaselineViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        m = config["model"]
        i = config["image"]

        self.img_to_tokens = ImageToTokens(
            i["img_size"],
            i["patch_size"],
            i["in_channels"],
            m["d_model"],
            i["use_class_token"],
            i["use_pos_encoding"]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m["d_model"],
            nhead=m["num_heads"],
            dim_feedforward=m["ff_dim"],
            dropout=m["dropout"],
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=m["num_layers"]
        )

        self.norm = nn.LayerNorm(m["d_model"])
        self.head = nn.Linear(m["d_model"], i["num_classes"])

        self.use_class_token = i["use_class_token"]

    def forward(self, x):
        x = self.img_to_tokens(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.use_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.head(x)