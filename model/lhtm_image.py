import torch.nn as nn
from model.image_embeddings import ImageToTokens
from model.encoder_block import EncoderBlock

class LHTMImageClassifier(nn.Module):
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

        self.layers = nn.ModuleList([
            EncoderBlock(
                m["d_model"],
                m["num_heads"],
                m["window_size"],
                m["ff_dim"],
                m["num_routing_tokens"],
                m["dropout"]
            )
            for _ in range(m["num_layers"])
        ])

        self.norm = nn.LayerNorm(m["d_model"])
        self.head = nn.Linear(m["d_model"], i["num_classes"])

        self.use_class_token = i["use_class_token"]

    def forward(self, x):
        x = self.img_to_tokens(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        if self.use_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.head(x)