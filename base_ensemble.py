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
BATCH_SIZE = 8
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


def build_model(name, num_classes):
    if name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        trainable = model.fc.parameters()

    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        trainable = model.classifier.parameters()

    elif name == "wideresnet50":
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        trainable = model.fc.parameters()

    else:
        raise ValueError(f"Unsupported model: {name}")

    for p in model.parameters():
        p.requires_grad = False
    for p in trainable:
        p.requires_grad = True

    return model


def build_models(model_names, num_classes, device):
    return {name: build_model(name, num_classes).to(device) for name in model_names}


@torch.no_grad()
def evaluate(nets, model_names, loader, criterion, device):
    for net in nets.values():
        net.eval()

    total = 0
    loss_sum = 0.0
    correct_ind = {name: 0 for name in model_names}
    correct_soft = 0
    correct_hard = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = {name: nets[name](x) for name in model_names}

        loss = sum(criterion(logits[name], y) for name in model_names)
        loss_sum += loss.item() * y.size(0)

        for name in model_names:
            pred = logits[name].argmax(dim=1)
            correct_ind[name] += (pred == y).sum().item()

        probs = [torch.softmax(logits[name], dim=1) for name in model_names]
        pred_soft = (sum(probs) / len(model_names)).argmax(dim=1)
        correct_soft += (pred_soft == y).sum().item()

        pred_stack = torch.stack([logits[name].argmax(dim=1) for name in model_names], dim=1)
        pred_hard, _ = torch.mode(pred_stack, dim=1)
        correct_hard += (pred_hard == y).sum().item()

        total += y.size(0)

    return {
        "loss": loss_sum / max(1, total),
        "individual": {name: 100.0 * correct_ind[name] / max(1, total) for name in model_names},
        "soft": 100.0 * correct_soft / max(1, total),
        "hard": 100.0 * correct_hard / max(1, total),
    }


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
    print(f"[{split_tag}] models={args.models}")

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

    nets = build_models(args.models, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    params = []
    for name in args.models:
        params.extend(p for p in nets[name].parameters() if p.requires_grad)

    optimizer = optim.AdamW(params, lr=LR, weight_decay=0.0)

    best_soft = -1.0
    best_states = None

    for epoch in range(1, NUM_EPOCHS + 1):
        for name in args.models:
            nets[name].train()

        total_loss = 0.0
        total_num = 0
        train_correct = {name: 0 for name in args.models}

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = {name: nets[name](x) for name in args.models}
            loss = sum(criterion(logits[name], y) for name in args.models)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_num += bs

            for name in args.models:
                train_correct[name] += (logits[name].argmax(dim=1) == y).sum().item()

        train_loss = total_loss / max(1, total_num)
        train_acc = {name: 100.0 * train_correct[name] / max(1, total_num) for name in args.models}

        val_result = evaluate(nets, args.models, val_loader, criterion, device)

        train_text = " / ".join(f"{name}:{train_acc[name]:.2f}" for name in args.models)
        val_text = " / ".join(f"{name}:{val_result['individual'][name]:.2f}" for name in args.models)

        print(
            f"[{split_tag}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_text} | "
            f"val_acc={val_text} | "
            f"soft={val_result['soft']:.2f} hard={val_result['hard']:.2f}"
        )

        if val_result["soft"] > best_soft:
            best_soft = val_result["soft"]
            best_states = {name: nets[name].state_dict() for name in args.models}

    os.makedirs(args.output_dir, exist_ok=True)
    if best_states is not None:
        for name in args.models:
            save_path = os.path.join(args.output_dir, f"{name}_{split_tag}_best.pth")
            torch.save(best_states[name], save_path)

    print(f"[{split_tag}] Best soft voting acc: {best_soft:.2f}")
    return best_soft


def main(args):
    seed_everything(SEED)

    if args.n_folds <= 1:
        train_one_split(args)
        return

    scores = []
    for fold_idx in range(args.n_folds):
        scores.append(train_one_split(args, fold_idx))

    print("\nCross-validation summary")
    print(f"Scores: {[round(x, 2) for x in scores]}")
    print(f"Mean: {np.mean(scores):.2f}")
    print(f"Std:  {np.std(scores):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--n_folds", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["resnet101", "densenet121", "wideresnet50"])
    args = parser.parse_args()

    main(args)