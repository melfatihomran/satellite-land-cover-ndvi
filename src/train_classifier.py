"""
train_classifier.py

Fine-tunes a pretrained ResNet-50 on the EuroSAT land cover dataset.

Same transfer-learning pattern used for a plant disease CNN/ResNet-50
project, applied here to satellite imagery instead of leaf photos:
  1. Load ImageNet-pretrained ResNet-50
  2. Replace the final FC layer for 10 EuroSAT classes
  3. Fine-tune end-to-end with a low learning rate
  4. Track train/val accuracy, save best checkpoint

Usage:
    python src/train_classifier.py --epochs 10 --batch-size 32
"""

import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report

from dataset_utils import get_eurosat_dataloaders


def build_model(num_classes: int = 10, freeze_backbone: bool = False):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return running_loss / total, running_correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on EuroSAT")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output", type=str, default="./models/resnet50_eurosat.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = get_eurosat_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    model = build_model(num_classes=len(classes), freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), args.output)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {args.output}")

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=classes))


if __name__ == "__main__":
    main()
