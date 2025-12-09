import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
data_dir = "dataset"
batch_size = 32
num_epochs = 20        # keep ≥ 20 as you asked
learning_rate = 5e-5
img_size = 128
max_images_per_class = 700
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------
# Improved augmentation (core change #1)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=20, translate=(0.10, 0.10), scale=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# -----------------------------
# Load dataset
# -----------------------------
print("[INFO] Loading dataset...")
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print(f"[INFO] Detected classes: {class_names}")

# Filter target classes
target_classes = ["sphere", "cube", "cone", "cylinder"]
target_indices = [i for i, (path, label) in enumerate(dataset.samples)
                  if dataset.classes[label] in target_classes]

samples_by_class = {cls: [] for cls in target_classes}
for idx in target_indices:
    path, label = dataset.samples[idx]
    cls_name = dataset.classes[label]
    samples_by_class[cls_name].append(idx)

# Balance dataset
balanced_indices = []
print("[INFO] Balancing dataset...")
for cls_name, idx_list in samples_by_class.items():
    idx_list = sorted(idx_list)[:max_images_per_class]
    balanced_indices.extend(idx_list)
    print(f"    Using {len(idx_list)} images for class '{cls_name}'")

balanced_dataset = Subset(dataset, balanced_indices)

# -----------------------------
# Train/Val Split
# -----------------------------
print("[INFO] Splitting dataset...")
indices = list(range(len(balanced_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_data = Subset(balanced_dataset, train_idx)
val_data = Subset(balanced_dataset, val_idx)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# -----------------------------
# Model Setup (core change #2 — EfficientNet-B0)
# -----------------------------
print("[INFO] Initializing EfficientNet-B0...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(target_classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Training & Evaluation
# -----------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# -----------------------------
# Training Loop
# -----------------------------
print("[INFO] Starting training...")
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_tactile_classifier.pth")
        print(f"[INFO] New best model saved at epoch {epoch+1} with Val Acc {val_acc*100:.2f}%")

print(f"[INFO] Training complete. Best validation accuracy: {best_acc*100:.2f}%")
