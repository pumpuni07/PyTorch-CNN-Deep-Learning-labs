"""
Logistic Regression: Cross Entropy Loss vs Bad Initialization
=============================================================
Course  : IBM AI Engineering Professional Certificate
Module  : Deep Neural Networks with PyTorch
Author  : Jack Pumpuni Frimpong-Manso
GitHub  : github.com/pumpuni07

Topics covered:
  - Binary logistic regression with sigmoid + BCELoss
  - Why Cross Entropy outperforms MSE for classification tasks
  - How bad weight initialization causes gradient saturation
  - Accuracy comparison: good init (~100%) vs bad init (~60%)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
print(f"PyTorch version: {torch.__version__}")

# =============================================================================
# DATASET AND MODEL
# =============================================================================

class BinaryData(Dataset):
    """Synthetic 1D binary classification dataset (step function at x=0)."""
    def __init__(self, n_samples=100):
        torch.manual_seed(1)
        self.x = torch.arange(-3, 3, 6 / n_samples).view(-1, 1).float()
        self.y = torch.zeros(n_samples, 1)
        self.y[n_samples // 2:] = 1.0
        self.len = n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class LogisticRegressionModel(nn.Module):
    """Single-layer logistic regression: sigmoid(wx + b)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Helpers
def train_model(model, criterion, optimizer, dataloader, epochs=100):
    """Train and return per-epoch average loss list."""
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_b, y_b in dataloader:
            optimizer.zero_grad()
            yhat = model(x_b)
            loss = criterion(yhat, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(dataloader))
    return history


def evaluate(model, dataset):
    """Return accuracy as a fraction in [0, 1]."""
    with torch.no_grad():
        yhat  = model(dataset.x)
        label = (yhat > 0.5).float()
        acc   = (label == dataset.y).float().mean().item()
    return acc


# =============================================================================
# DATA SETUP
# =============================================================================

print("\n" + "="*60)
print("DATA SETUP")
print("="*60)

data_set   = BinaryData(n_samples=100)
trainloader = DataLoader(dataset=data_set, batch_size=10, shuffle=True)

print(f"Samples: {len(data_set)}")
print(f"Class 0: {(data_set.y == 0).sum().item()},  Class 1: {(data_set.y == 1).sum().item()}")

# =============================================================================
# CROSS ENTROPY (GOOD INITIALIZATION)
# =============================================================================

print("\n" + "="*60)
print("CROSS ENTROPY — GOOD INITIALIZATION")
print("="*60)

torch.manual_seed(0)
model_good    = LogisticRegressionModel()
criterion_bce = nn.BCELoss()
opt_good      = torch.optim.SGD(model_good.parameters(), lr=2.0)

print(f"Initial w = {model_good.linear.weight.item():.4f}, "
      f"b = {model_good.linear.bias.item():.4f}")

loss_good = train_model(model_good, criterion_bce, opt_good, trainloader, epochs=100)
acc_good  = evaluate(model_good, data_set)

print(f"Final accuracy: {acc_good * 100:.1f}%")
print(f"Final BCE loss: {loss_good[-1]:.4f}")

# =============================================================================
# CROSS ENTROPY (BAD INITIALIZATION)
# =============================================================================

print("\n" + "="*60)
print("CROSS ENTROPY — BAD INITIALIZATION")
print("="*60)

torch.manual_seed(0)
model_bad = LogisticRegressionModel()

# Force large weights → sigmoid saturated from epoch 1
model_bad.state_dict()['linear.weight'][0] = torch.tensor([-15.0])
model_bad.state_dict()['linear.bias'][0]   = torch.tensor([3.0])

print(f"Initial w = {model_bad.linear.weight.item():.4f}, "
      f"b = {model_bad.linear.bias.item():.4f}")

# Show saturation effect
x_probe = data_set.x[[0, 49, 99]]
with torch.no_grad():
    sig_probe = model_bad(x_probe).flatten().tolist()
print(f"Sigmoid outputs before training (x={x_probe.flatten().tolist()}):")
print(f"  {[f'{v:.6f}' for v in sig_probe]}")
print("  → Near 0 or 1 already — gradients effectively zero!")

opt_bad  = torch.optim.SGD(model_bad.parameters(), lr=2.0)
loss_bad = train_model(model_bad, criterion_bce, opt_bad, trainloader, epochs=100)
acc_bad  = evaluate(model_bad, data_set)

print(f"Final accuracy: {acc_bad * 100:.1f}%")
print(f"Final BCE loss: {loss_bad[-1]:.4f}")

# =============================================================================
# SIDE-BY-SIDE COMPARISON PLOTS
# =============================================================================

print("\n" + "="*60)
print("GENERATING COMPARISON PLOTS")
print("="*60)

mask0 = (data_set.y[:, 0] == 0)
mask1 = (data_set.y[:, 0] == 1)
x_plot = torch.linspace(-3, 3, 300).unsqueeze(1)

with torch.no_grad():
    y_good_plot = model_good(x_plot).numpy()
    y_bad_plot  = model_bad(x_plot).numpy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Training loss
axes[0].plot(loss_good, 'b-',  linewidth=2, label=f'Good init ({acc_good*100:.0f}% acc)')
axes[0].plot(loss_bad,  'r--', linewidth=2, label=f'Bad init  ({acc_bad*100:.0f}% acc)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].set_title('Training Loss Curves', fontsize=12, fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Decision boundaries
axes[1].scatter(data_set.x[mask0].numpy(), data_set.y[mask0].numpy(),
                color='blue', alpha=0.5, edgecolors='k', s=30, label='Class 0')
axes[1].scatter(data_set.x[mask1].numpy(), data_set.y[mask1].numpy(),
                color='red',  alpha=0.5, edgecolors='k', s=30, label='Class 1')
axes[1].plot(x_plot.numpy(), y_good_plot, 'b-',  linewidth=2.5, label='Good init')
axes[1].plot(x_plot.numpy(), y_bad_plot,  'r--', linewidth=2.5, label='Bad init')
axes[1].axhline(0.5, color='gray', linestyle=':', label='Threshold 0.5')
axes[1].set_xlabel('x'); axes[1].set_ylabel('P(class=1)')
axes[1].set_title('Learned Sigmoid Curves', fontsize=12, fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Plot 3: Gradient magnitude BCE vs MSE
p_vals  = np.linspace(0.01, 0.99, 200)
bce_grad = np.abs(-(1 / p_vals))          # |∂BCE/∂p| for y=1
mse_grad = np.abs(2 * (p_vals - 1))       # |∂MSE/∂p| for y=1
axes[2].plot(p_vals, bce_grad, 'b-',  linewidth=2.5, label='|∂BCE/∂p| (y=1)')
axes[2].plot(p_vals, mse_grad, 'r--', linewidth=2.5, label='|∂MSE/∂p| (y=1)')
axes[2].set_xlabel('Predicted p'); axes[2].set_ylabel('|Gradient|')
axes[2].set_title('BCE vs MSE: Gradient Magnitude\n(larger = stronger training signal)',
                  fontsize=11, fontweight='bold')
axes[2].set_ylim(0, 10); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle('Logistic Regression: Cross Entropy vs Bad Initialization',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('logistic_regression_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: logistic_regression_comparison.png")

# =============================================================================
# RESULTS SUMMARY TABLE
# =============================================================================

print("\n" + "="*55)
print("FINAL RESULTS SUMMARY")
print("="*55)
print(f"{'Metric':<30} {'Good Init':>10} {'Bad Init':>10}")
print("-"*55)
print(f"{'Accuracy':<30} {acc_good*100:>9.1f}% {acc_bad*100:>9.1f}%")
print(f"{'Final BCE Loss':<30} {loss_good[-1]:>10.4f} {loss_bad[-1]:>10.4f}")
print(f"{'Epoch 1 Loss':<30} {loss_good[0]:>10.4f} {loss_bad[0]:>10.4f}")
print(f"{'Epoch 10 Loss':<30} {loss_good[9]:>10.4f} {loss_bad[9]:>10.4f}")
print()
print("Key takeaway: Bad initialization saturates the sigmoid at epoch 1,")
print("producing near-zero gradients. The model barely improves over 100 epochs.")
print("Good initialization keeps gradients healthy, reaching high accuracy quickly.")
