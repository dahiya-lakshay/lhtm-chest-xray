import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os


def get_dataloader(data_path, batch_size):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train = ImageFolder(os.path.join(data_path, "train"), transform=transform)
    val = ImageFolder(os.path.join(data_path, "val"), transform=transform)
    test = ImageFolder(os.path.join(data_path, "test"), transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, config, dataset, data_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model.to(device)

    train_loader, val_loader, test_loader = get_dataloader(
        data_path, config["training"]["batch_size"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    best_loss = float("inf")
    patience = 5
    counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config["training"]["epochs"]):

        model.train()
        running_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)

                loss = criterion(out, y)
                val_loss += loss.item()

                probs = torch.softmax(out, dim=1)[:, 1]
                _, pred = out.max(1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)

        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, Acc={val_acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    # Plot graphs
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")

    plt.clf()
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_curve.png")