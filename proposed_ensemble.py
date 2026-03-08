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


SEED = "seed"
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 4

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PKLDataset(Dataset):
    def __init__(self, data_list, class_to_idx, transform=None):
        self.data_list = data_list
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def _to_pil_rgb(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3:
                return Image.fromarray(image).convert("RGB")
            raise ValueError(f"Unsupported ndarray shape: {image.shape}")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        raise ValueError(f"Unsupported image type: {type(image)}")

    def __getitem__(self, idx):
        image, label = self.data_list[idx]
        image = self._to_pil_rgb(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(self.class_to_idx[label], dtype=torch.long)


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


def resolve_ckpt_paths(ckpt_dir):
    ckpt_paths = {
        "resnet101": os.path.join(ckpt_dir, "resnet101.pth"),
        "densenet121": os.path.join(ckpt_dir, "densenet121.pth"),
        "wideresnet50": os.path.join(ckpt_dir, "wideresnet50.pth"),
    }

    for model_name, path in ckpt_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {path}")

    return ckpt_paths


def build_backbone_arch(model_name: str, num_classes: int):
    if model_name == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "wideresnet50":
        model = models.wide_resnet50_2(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model


def load_frozen_individual(model_name: str, ckpt_path: str, num_classes: int, device):
    model = build_backbone_arch(model_name, num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)

    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    model.eval()
    return model


def norm_logits(x, eps=1e-6):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)


class LearnedEnsemble(nn.Module):
    def __init__(self, resnet, densenet, wideresnet, combiner_type="conv1d"):
        super().__init__()
        self.resnet = resnet
        self.densenet = densenet
        self.wideresnet = wideresnet
        self.combiner_type = combiner_type

        if combiner_type == "conv1d":
            self.combiner = nn.Conv1d(3, 1, kernel_size=1, bias=True)
        elif combiner_type == "linear":
            self.combiner = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(f"Unknown combiner_type: {combiner_type}")

    def forward(self, x):
        r = norm_logits(self.resnet(x))
        d = norm_logits(self.densenet(x))
        w = norm_logits(self.wideresnet(x))

        z = torch.stack([r, d, w], dim=2)  # (B, C, 3)

        if self.combiner_type == "linear":
            b, c, _ = z.shape
            z = z.reshape(b * c, 3)
            out = self.combiner(z).reshape(b, c)
        else:
            z = z.permute(0, 2, 1)  # (B, 3, C)
            out = self.combiner(z).squeeze(1)

        return out


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

    return total_loss / total_num, 100.0 * total_correct / total_num


def train_one_split(args, fold_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_tag = "single_split" if fold_idx is None else f"fold_{fold_idx}"

    train_path, val_path = resolve_split_paths(args.data_dir, fold_idx)
    ckpt_paths = resolve_ckpt_paths(args.ckpt_dir)

    print(f"\n[{split_tag}] device={device}")
    print(f"[{split_tag}] train={train_path}")
    print(f"[{split_tag}] val={val_path}")
    print(f"[{split_tag}] ckpt_dir={args.ckpt_dir}")
    print(f"[{split_tag}] combiner={args.combiner}")

    train_list = load_pkl(train_path)
    val_list = load_pkl(val_path)

    class_to_idx = build_class_to_idx(train_list, val_list)
    num_classes = len(class_to_idx)
    print(f"[{split_tag}] num_classes={num_classes}")

    transform = build_transform()

    train_loader = DataLoader(
        PKLDataset(train_list, class_to_idx, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PKLDataset(val_list, class_to_idx, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    resnet = load_frozen_individual("resnet101", ckpt_paths["resnet101"], num_classes, device)
    densenet = load_frozen_individual("densenet121", ckpt_paths["densenet121"], num_classes, device)
    wideresnet = load_frozen_individual("wideresnet50", ckpt_paths["wideresnet50"], num_classes, device)

    model = LearnedEnsemble(
        resnet=resnet,
        densenet=densenet,
        wideresnet=wideresnet,
        combiner_type=args.combiner,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.combiner.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"best_ensemble_{args.combiner}_{split_tag}.pth")

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

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_num += y.size(0)

        train_loss = total_loss / total_num
        train_acc = 100.0 * total_correct / total_num
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
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--combiner", type=str, default="conv1d", choices=["conv1d", "linear"])
    parser.add_argument("--n_folds", type=int, default=1)

    args = parser.parse_args()
    main(args)