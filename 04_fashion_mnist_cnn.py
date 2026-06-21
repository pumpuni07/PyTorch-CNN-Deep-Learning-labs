"""
Fashion-MNIST Classification: CNN vs CNN with Batch Normalization
=================================================================
Course  : IBM AI Engineering Professional Certificate
Module  : Deep Neural Networks with PyTorch
Author  : Jack Pumpuni Frimpong-Manso
GitHub  : github.com/pumpuni07
Dataset : https://github.com/zalandoresearch/fashion-mnist

Topics covered:
  - Custom Dataset class with Resize + ToTensor transforms
  - Two-layer CNN architecture (Conv→ReLU→Pool ×2 → FC)
  - CNN_batch: same architecture with BatchNorm2d and BatchNorm1d
  - Training with SGD + CrossEntropyLoss
  - Side-by-side comparison of cost, accuracy, and per-class breakdown
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

torch.manual_seed(0)
print(f"PyTorch version: {torch.__version__}")

# =============================================================================
# CONSTANTS
# =============================================================================

IMAGE_SIZE = 16  # Resize 28×28 → 16×16
BATCH_SIZE = 100
N_EPOCHS   = 5
LR         = 0.1

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# =============================================================================
# DATASET
# =============================================================================

print("\n" + "="*60)
print("DATASET SETUP")
print("="*60)

composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset_train = dsets.FashionMNIST(
    root='.fashion/data', train=True,  transform=composed, download=True
)
dataset_val = dsets.FashionMNIST(
    root='.fashion/data', train=False, transform=composed, download=True
)

print(f"Training samples:   {len(dataset_train):,}")
print(f"Validation samples: {len(dataset_val):,}")
print(f"Image tensor shape: {dataset_train[0][0].shape}")

train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset=dataset_val,   batch_size=BATCH_SIZE, shuffle=False)

# Visualise samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    img, label = dataset_val[i]
    ax.imshow(img.numpy().squeeze(), cmap='gray')
    ax.set_title(f'{CLASS_NAMES[label]}\n(y={label})', fontsize=9)
    ax.axis('off')
plt.suptitle('Fashion-MNIST Validation Samples', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fashion_mnist_samples.png")

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CNN(nn.Module):
    """Standard two-layer CNN for Fashion-MNIST (10-class classification)."""

    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN, self).__init__()
        self.cnn1     = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2     = nn.Conv2d(out_1, out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1      = nn.Linear(out_2 * 4 * 4, number_of_classes)

    def forward(self, x):
        x = torch.relu(self.cnn1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.cnn2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class CNN_batch(nn.Module):
    """
    Two-layer CNN with Batch Normalization after each conv and FC layer.

    BatchNorm2d(C): normalises each channel across batch and spatial dims.
    BatchNorm1d(C): normalises each feature across the batch.
    """

    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1     = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2     = nn.Conv2d(out_1, out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1      = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1   = nn.BatchNorm1d(number_of_classes)

    def forward(self, x):
        x = torch.relu(self.conv1_bn(self.cnn1(x)))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2_bn(self.cnn2(x)))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        return self.bn_fc1(self.fc1(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_and_evaluate(model, train_loader, test_loader, dataset_val,
                       n_epochs=5, lr=0.1, model_name='Model'):
    """Train with SGD + CrossEntropyLoss. Returns (cost_list, accuracy_list)."""
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.SGD(model.parameters(), lr=lr)
    N_test        = len(dataset_val)
    cost_list     = []
    accuracy_list = []
    start_time    = time.time()

    print(f"\nTraining {model_name} ({count_params(model):,} params)...")
    for epoch in range(n_epochs):
        model.train()
        cost = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            z    = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            cost += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x_t, y_t in test_loader:
                z = model(x_t)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_t).sum().item()

        acc = correct / N_test
        cost_list.append(cost)
        accuracy_list.append(acc)
        print(f"  Epoch {epoch+1}/{n_epochs} | "
              f"Cost: {cost:.2f} | "
              f"Val Acc: {acc*100:.2f}% | "
              f"{time.time()-start_time:.1f}s elapsed")

    return cost_list, accuracy_list


# =============================================================================
# TRAIN BOTH MODELS
# =============================================================================

torch.manual_seed(0)
model_cnn   = CNN(out_1=16, out_2=32, number_of_classes=10)
cost_cnn, acc_cnn = train_and_evaluate(
    model_cnn, train_loader, test_loader, dataset_val,
    n_epochs=N_EPOCHS, lr=LR, model_name='CNN'
)

torch.manual_seed(0)
model_batch   = CNN_batch(out_1=16, out_2=32, number_of_classes=10)
cost_batch, acc_batch = train_and_evaluate(
    model_batch, train_loader, test_loader, dataset_val,
    n_epochs=N_EPOCHS, lr=LR, model_name='CNN_batch'
)

# =============================================================================
# COMPARISON PLOTS
# =============================================================================

print("\nGenerating comparison plots...")
epochs = range(1, N_EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Cost
axes[0].plot(epochs, cost_cnn,   'b-o', linewidth=2, markersize=8, label='CNN')
axes[0].plot(epochs, cost_batch, 'r-s', linewidth=2, markersize=8, label='CNN+BatchNorm')
axes[0].set_xlabel('Epoch', fontsize=12); axes[0].set_ylabel('Training Cost', fontsize=12)
axes[0].set_title('Training Cost per Epoch', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=12); axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs, [a*100 for a in acc_cnn],
             'b-o', linewidth=2, markersize=8,
             label=f'CNN (final: {acc_cnn[-1]*100:.2f}%)')
axes[1].plot(epochs, [a*100 for a in acc_batch],
             'r-s', linewidth=2, markersize=8,
             label=f'CNN+BN (final: {acc_batch[-1]*100:.2f}%)')
axes[1].set_xlabel('Epoch', fontsize=12); axes[1].set_ylabel('Val Accuracy (%)', fontsize=12)
axes[1].set_title('Validation Accuracy per Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].legend(fontsize=12); axes[1].grid(True, alpha=0.3)

plt.suptitle('Fashion-MNIST: CNN vs CNN with Batch Normalization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fashion_mnist_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fashion_mnist_results.png")

# =============================================================================
# PER-CLASS ACCURACY
# =============================================================================

def per_class_accuracy(model, dataset_val, n_classes=10):
    model.eval()
    correct = torch.zeros(n_classes)
    total   = torch.zeros(n_classes)
    loader  = DataLoader(dataset_val, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            z = model(x)
            _, preds = torch.max(z, 1)
            for c in range(n_classes):
                mask = (y == c)
                correct[c] += (preds[mask] == y[mask]).sum().item()
                total[c]   += mask.sum().item()
    return {CLASS_NAMES[c]: (correct[c] / total[c]).item() * 100 for c in range(n_classes)}


class_acc_cnn   = per_class_accuracy(model_cnn,   dataset_val)
class_acc_batch = per_class_accuracy(model_batch, dataset_val)

print("\nPer-class accuracy:")
print(f"{'Class':<18} {'CNN':>8} {'CNN+BN':>10}")
print("-" * 40)
for cls in CLASS_NAMES:
    print(f"{cls:<18} {class_acc_cnn[cls]:>7.1f}%  {class_acc_batch[cls]:>7.1f}%")

# Bar chart
x_pos = np.arange(len(CLASS_NAMES))
width = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x_pos - width/2, [class_acc_cnn[c] for c in CLASS_NAMES],
       width, label='CNN', color='steelblue', edgecolor='k')
ax.bar(x_pos + width/2, [class_acc_batch[c] for c in CLASS_NAMES],
       width, label='CNN+BatchNorm', color='tomato', edgecolor='k')
ax.set_xticks(x_pos)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Accuracy: CNN vs CNN+BatchNorm', fontsize=13, fontweight='bold')
ax.legend(fontsize=12); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fashion_mnist_per_class.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fashion_mnist_per_class.png")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

print("\n" + "="*55)
print("FINAL RESULTS: Fashion-MNIST CNN Comparison")
print("="*55)
print(f"{'Metric':<30} {'CNN':>10} {'CNN+BN':>10}")
print("-"*55)
print(f"{'Parameters':<30} {count_params(model_cnn):>10,} {count_params(model_batch):>10,}")
print(f"{'Final Val Accuracy':<30} {acc_cnn[-1]*100:>9.2f}% {acc_batch[-1]*100:>9.2f}%")
print(f"{'Final Training Cost':<30} {cost_cnn[-1]:>10.2f} {cost_batch[-1]:>10.2f}")
print(f"{'Epoch 1 Accuracy':<30} {acc_cnn[0]*100:>9.2f}% {acc_batch[0]*100:>9.2f}%")
winner = 'CNN+BatchNorm' if acc_batch[-1] >= acc_cnn[-1] else 'CNN'
print(f"\n✓ Best performing model: {winner}")
