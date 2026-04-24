import yaml
import kagglehub

from model.lhtm_image import LHTMImageClassifier
from model.baseline_vit import BaselineViT
from model.cnn_baseline import CNNBaseline
from training.train import train

if __name__ == "__main__":

    # Download dataset
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Dataset Path:", path)

    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    # Chest X-ray config
    config["image"]["in_channels"] = 3
    config["image"]["num_classes"] = 2
    config["image"]["img_size"] = 224

    dataset = "chest_xray"

    print("\n--- CNN Baseline ---")
    cnn = CNNBaseline(num_classes=2, in_channels=3)
    train(cnn, config, dataset, path)

    print("\n--- ViT Baseline ---")
    vit = BaselineViT(config)
    train(vit, config, dataset, path)

    print("\n--- LHTM ---")
    lhtm = LHTMImageClassifier(config)
    train(lhtm, config, dataset, path)