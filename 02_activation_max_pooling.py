"""
Activation Functions and Max Pooling — PyTorch Implementation
=============================================================
Course  : IBM AI Engineering Professional Certificate
Module  : Deep Neural Networks with PyTorch
Author  : Jack Pumpuni Frimpong-Manso
GitHub  : github.com/pumpuni07

Topics covered:
  - ReLU activation: max(0, x) applied element-wise to feature maps
  - Sigmoid and Tanh for comparison
  - Max Pooling with stride control (overlapping vs non-overlapping)
  - Full pipeline: Conv2d → ReLU → MaxPool2d with shape tracking
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}")

# =============================================================================
# 1. ACTIVATION FUNCTIONS
# =============================================================================

print("\n" + "="*60)
print("1. ACTIVATION FUNCTIONS")
print("="*60)

# Build Conv layer with Sobel-X kernel
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
Gx = torch.tensor([[1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0]])
conv.state_dict()['weight'][0][0] = Gx
conv.state_dict()['bias'][0] = 0.0

# Input image with vertical edge
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1
print("Input image (5×5):")
print(image[0, 0])

# Convolution
Z = conv(image)
print("\nActivation map Z (post-Conv, pre-ReLU):")
print(Z[0, 0].detach())

# ReLU — two equivalent approaches
A_functional = torch.relu(Z)
relu_module   = nn.ReLU()
A_module      = relu_module(Z)

print("\nAfter torch.relu(Z):")
print(A_functional[0, 0].detach())
print("\nAfter nn.ReLU()(Z) — identical:")
print(A_module[0, 0].detach())

assert torch.allclose(A_functional, A_module), "Results should be identical!"
print("✓ Both approaches produce the same result.")

# Plot activation functions
x_vals = torch.linspace(-4, 4, 200)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

fns = [
    (torch.relu(x_vals),    'ReLU: max(0, x)',           'blue'),
    (torch.sigmoid(x_vals), 'Sigmoid: 1/(1+e⁻ˣ)',        'red'),
    (torch.tanh(x_vals),    'Tanh: (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)', 'green'),
]
for ax, (y, label, color) in zip(axes, fns):
    ax.plot(x_vals.numpy(), y.numpy(), color=color, linewidth=2)
    ax.set_title(label, fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)

plt.suptitle('Activation Functions Used in CNNs', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: activation_functions.png")

# =============================================================================
# 2. MAX POOLING
# =============================================================================

print("\n" + "="*60)
print("2. MAX POOLING")
print("="*60)

# Create a 4×4 feature map with known values
image1 = torch.zeros(1, 1, 4, 4)
image1[0, 0, 0, :] = torch.tensor([1.0,  2.0, 3.0, -4.0])
image1[0, 0, 1, :] = torch.tensor([0.0,  2.0, -3.0,  0.0])
image1[0, 0, 2, :] = torch.tensor([0.0,  2.0,  3.0,  1.0])
image1[0, 0, 3, :] = torch.tensor([0.0,  1.0,  1.0,  0.0])

print("Input feature map (4×4):")
print(image1[0, 0])

# Max Pool with stride=1 (overlapping)
max_s1 = nn.MaxPool2d(2, stride=1)
r1 = max_s1(image1)
print("\nMaxPool2d(kernel=2, stride=1) — overlapping:")
print(r1[0, 0].detach())
print(f"Output shape: {r1.shape[2:]}  (floor((4-2)/1)+1 = 3)")

# Max Pool with stride=2 (non-overlapping, default)
max_s2 = nn.MaxPool2d(2)   # stride defaults to kernel_size
r2 = max_s2(image1)
print("\nMaxPool2d(kernel=2, stride=2) — non-overlapping:")
print(r2[0, 0].detach())
print(f"Output shape: {r2.shape[2:]}  (floor((4-2)/2)+1 = 2)")

# Manual verification of non-overlapping windows
print("\nManual verification (non-overlapping, 2×2 windows):")
x = image1[0, 0]
print(f"  Top-left   [0:2, 0:2]: {x[0:2, 0:2].flatten().tolist()}  → max = {x[0:2, 0:2].max().item()}")
print(f"  Top-right  [0:2, 2:4]: {x[0:2, 2:4].flatten().tolist()} → max = {x[0:2, 2:4].max().item()}")
print(f"  Bot-left   [2:4, 0:2]: {x[2:4, 0:2].flatten().tolist()}  → max = {x[2:4, 0:2].max().item()}")
print(f"  Bot-right  [2:4, 2:4]: {x[2:4, 2:4].flatten().tolist()}  → max = {x[2:4, 2:4].max().item()}")

# =============================================================================
# 3. FULL PIPELINE: Conv → ReLU → MaxPool
# =============================================================================

print("\n" + "="*60)
print("3. FULL CNN BLOCK: Conv → ReLU → MaxPool")
print("="*60)

torch.manual_seed(0)

# 7×7 image, vertical edge at col 3
image_full = torch.zeros(1, 1, 7, 7)
image_full[0, 0, :, 3] = 1.0

# Layers
conv_layer = nn.Conv2d(1, 1, kernel_size=3, padding=0)
conv_layer.state_dict()['weight'][0][0] = torch.tensor([[ 1.0,  0.0, -1.0],
                                                         [ 2.0,  0.0, -2.0],
                                                         [ 1.0,  0.0, -1.0]])
conv_layer.state_dict()['bias'][0] = 0.0
relu_layer = nn.ReLU()
pool_layer = nn.MaxPool2d(2, stride=2)

# Forward
Z_full = conv_layer(image_full)
A_full = relu_layer(Z_full)
P_full = pool_layer(A_full)

print(f"Input:       {image_full.shape}")
print(f"After Conv:  {Z_full.shape}   (7-3+1=5)")
print(f"After ReLU:  {A_full.shape}   (unchanged)")
print(f"After Pool:  {P_full.shape}   (floor((5-2)/2)+1=2)")

# Spatial progression chart
print("\nSpatial dimension summary:")
print(f"  {'Stage':<20} {'H×W'}")
print("  " + "-"*30)
stages = [
    ("Input",          image_full.shape[2:]),
    ("After Conv2d",   Z_full.shape[2:]),
    ("After ReLU",     A_full.shape[2:]),
    ("After MaxPool",  P_full.shape[2:]),
]
for name, shape in stages:
    print(f"  {name:<20} {shape[0]}×{shape[1]}")

# Plot pipeline
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
data_list = [
    (image_full[0, 0].numpy(), 'Input (7×7)', 'gray'),
    (Z_full[0, 0].detach().numpy(), 'After Conv (5×5)', 'RdBu'),
    (A_full[0, 0].detach().numpy(), 'After ReLU (5×5)', 'Greens'),
    (P_full[0, 0].detach().numpy(), 'After MaxPool (2×2)', 'YlOrRd'),
]
for ax, (data, title, cmap) in zip(axes, data_list):
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=10)
    H, W = data.shape
    for i in range(H):
        for j in range(W):
            ax.text(j, i, f'{data[i,j]:.0f}',
                    ha='center', va='center', fontsize=10,
                    color='black')
    plt.colorbar(im, ax=ax)

plt.suptitle('CNN Block: Conv → ReLU → MaxPool', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_block_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: cnn_block_pipeline.png")

# =============================================================================
# 4. SUMMARY TABLE
# =============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Component':<18} {'Operation':<35} {'Output Size Change'}")
print("-"*70)
print(f"{'nn.Conv2d':<18} {'Element-wise mult + sum (kernel slides)':<35} Shrinks by K-1 (no padding)")
print(f"{'nn.ReLU':<18} {'max(0, x) element-wise':<35} Unchanged")
print(f"{'nn.MaxPool2d':<18} {'Max value in each K×K window':<35} Shrinks by factor ~s")
