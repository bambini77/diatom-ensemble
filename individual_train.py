import os
import pickle
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


SEED = "your seed"
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PKLDataset(Dataset):
    def __init__(self, data_list, class_to_idx, transform):
        self.data_list = data_list
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, label = self.data_list[idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image = self.transform(image)
        target = self.class_to_idx[label]
        return image, torch.tensor(target, dtype=torch.long)


def build_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_class_to_idx(train_list, val_list):
    classes = sorted({label for _, label in train_list + val_list})
    return {c: i for i, c in enumerate(classes)}


def resolve_split_paths(data_dir, fold_idx=None):
    if fold_idx is None:
        train_path = os.path.join(data_dir, "train.pkl")
        val_path = os.path.join(data_dir, "val.pkl")
    else:
        train_path = os.path.join(data_dir, f"fold_{fold_idx}", "train.pkl")
        val_path = os.path.join(data_dir, f"fold_{fold_idx}", "val.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing val split: {val_path}")

    return train_path, val_path


def build_model(num_classes):
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_num += y.size(0)

    return total_loss / max(1, total_num), 100.0 * total_correct / max(1, total_num)


def train_one_split(args, fold_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_tag = "single_split" if fold_idx is None else f"fold_{fold_idx}"

    train_path, val_path = resolve_split_paths(args.data_dir, fold_idx)
    train_list = load_pkl(train_path)
    val_list = load_pkl(val_path)

    class_to_idx = build_class_to_idx(train_list, val_list)
    num_classes = len(class_to_idx)

    print(f"\n[{split_tag}] device={device}")
    print(f"[{split_tag}] train={train_path}")
    print(f"[{split_tag}] val={val_path}")
    print(f"[{split_tag}] num_classes={num_classes}")

    transform = build_transform()

    train_loader = DataLoader(
        PKLDataset(train_list, class_to_idx, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PKLDataset(val_list, class_to_idx, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"best_model_{split_tag}.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_num = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_num += bs

        train_loss = total_loss / max(1, total_num)
        train_acc = 100.0 * total_correct / max(1, total_num)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[{split_tag}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"[{split_tag}] Best val acc: {best_val_acc:.2f}")
    return best_val_acc


def main(args):
    seed_everything(SEED)

    if args.n_folds <= 1:
        train_one_split(args)
        return

    scores = []
    for fold_idx in range(args.n_folds):
        print(f"\n===== Running fold {fold_idx}/{args.n_folds - 1} =====")
        scores.append(train_one_split(args, fold_idx))

    print("\n===== Cross-validation summary =====")
    print(f"Fold scores: {[round(x, 2) for x in scores]}")
    print(f"Mean acc: {np.mean(scores):.2f}")
    print(f"Std acc:  {np.std(scores):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--n_folds", type=int, default=1)
    args = parser.parse_args()

    main(args)