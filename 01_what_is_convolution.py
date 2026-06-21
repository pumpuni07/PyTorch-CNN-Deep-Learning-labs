"""
What is Convolution? — PyTorch Implementation
==============================================
Course  : IBM AI Engineering Professional Certificate
Module  : Deep Neural Networks with PyTorch
Author  : Jack Pumpuni Frimpong-Manso
GitHub  : github.com/pumpuni07

Topics covered:
  - Convolution as a linear operation
  - Output size formula: M_new = floor((M + 2p - K) / s) + 1
  - Effect of stride on output dimensions
  - Zero padding to control spatial resolution
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}")

# =============================================================================
# 1. CONVOLUTION DEFINED
# =============================================================================

print("\n" + "="*60)
print("1. CONVOLUTION DEFINED")
print("="*60)

# Create Conv2d layer with a Sobel-X kernel (vertical edge detector)
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv.state_dict()['weight'][0][0] = torch.tensor([[ 1.0,  0.0, -1.0],
                                                    [ 2.0,  0.0, -2.0],
                                                    [ 1.0,  0.0, -1.0]])
conv.state_dict()['bias'][0] = 0.0

print("Kernel (Sobel-X — detects vertical edges):")
print(conv.state_dict()['weight'][0][0])

# Create a 5×5 dummy image with a vertical edge at column 2
# Shape: (N=1, C=1, H=5, W=5)
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1
print("\nInput image (5×5), vertical edge at column 2:")
print(image[0, 0])

# Apply convolution
z = conv(image)
print("\nOutput feature map after Sobel-X convolution:")
print(z[0, 0].detach())
print(f"Output shape: {z.shape}  →  (N=1, C_out=1, H_out=3, W_out=3)")

# Visualise
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(image[0, 0].numpy(), cmap='gray')
axes[0].set_title('Input Image (5×5)')
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f'{image[0,0,i,j].item():.0f}',
                     ha='center', va='center', color='red', fontsize=12)

kernel_np = conv.state_dict()['weight'][0][0].detach().numpy()
axes[1].imshow(kernel_np, cmap='RdBu', vmin=-2, vmax=2)
axes[1].set_title('Sobel-X Kernel (3×3)')
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{kernel_np[i,j]:.0f}',
                     ha='center', va='center', color='black', fontsize=14, fontweight='bold')

output_np = z[0, 0].detach().numpy()
axes[2].imshow(output_np, cmap='RdBu')
axes[2].set_title('Output Feature Map (3×3)')
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f'{output_np[i,j]:.0f}',
                     ha='center', va='center', color='black', fontsize=11)

plt.suptitle('Convolution: Input → Kernel → Output', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('convolution_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: convolution_demo.png")

# =============================================================================
# 2. DETERMINING THE SIZE OF OUTPUT
# =============================================================================

print("\n" + "="*60)
print("2. OUTPUT SIZE: M_new = M - K + 1")
print("="*60)

def conv_output_size(M: int, K: int, s: int = 1, p: int = 0) -> int:
    """
    Calculate the output spatial dimension of a convolution.

    Args:
        M : Input size (height or width, assuming square)
        K : Kernel size
        s : Stride (default 1)
        p : Zero padding (default 0)

    Returns:
        Output size as an integer (PyTorch floors non-integer results)
    """
    return (M + 2 * p - K) // s + 1

K = 2
M = 4
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K)
conv1.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0],
                                                    [1.0, 1.0]])
conv1.state_dict()['bias'][0] = 0.0

image1 = torch.ones(1, 1, M, M)
z1 = conv1(image1)

print(f"Input size M={M}, Kernel K={K}, Stride s=1, Padding p=0")
print(f"Formula:  M_new = {M} - {K} + 1 = {M - K + 1}")
print(f"Actual output shape: {z1.shape[2:]}  ✓")
print(f"\nOutput values:\n{z1.detach()}")

# =============================================================================
# 3. STRIDE PARAMETER
# =============================================================================

print("\n" + "="*60)
print("3. STRIDE PARAMETER")
print("="*60)

conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
conv3.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0],
                                                    [1.0, 1.0]])
conv3.state_dict()['bias'][0] = 0.0

M, K, s = 4, 2, 2
expected = conv_output_size(M, K, s)
z3 = conv3(image1)

print(f"Input M={M}, Kernel K={K}, Stride s={s}")
print(f"Formula: floor(({M}-{K})/{s}) + 1 = {expected}")
print(f"Actual shape: {z3.shape[2:]}  ✓")
print(f"\nOutput values:\n{z3.detach()}")

# Stride comparison table
print("\nStride comparison (M=8, K=3):")
print(f"{'Stride':>8} | {'Output Size':>12}")
print("-" * 25)
for s_val in [1, 2, 3, 4]:
    size = conv_output_size(8, 3, s_val)
    print(f"{s_val:>8} | {size:>12}")

# =============================================================================
# 4. ZERO PADDING
# =============================================================================

print("\n" + "="*60)
print("4. ZERO PADDING")
print("="*60)

# Without padding — stride=3 gives non-integer (floored by PyTorch)
K, s, M = 2, 3, 4
conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K, stride=s)
conv4.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0],
                                                    [1.0, 1.0]])
conv4.state_dict()['bias'][0] = 0.0
z4 = conv4(image1)
raw = (M - K) / s + 1
print(f"No padding: ({M}-{K})/{s} + 1 = {raw:.4f}  → floored to {int(raw)}")
print(f"Shape: {z4.shape[2:]}")

# With padding = 1
padding = 1
conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K, stride=s, padding=padding)
conv5.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0],
                                                    [1.0, 1.0]])
conv5.state_dict()['bias'][0] = 0.0
z5 = conv5(image1)
new_size = conv_output_size(M, K, s, padding)
print(f"\nWith padding={padding}: M' = {M}+2×{padding}={M+2*padding}, "
      f"output = ({M+2*padding}-{K})//{s}+1 = {new_size}")
print(f"Shape: {z5.shape[2:]}  ✓")

# Full comparison table
print("\nPadding & Stride comparison (M=6, K=3):")
M_f, K_f = 6, 3
print(f"{'Stride':>8} | {'Padding':>8} | {'Output':>8}")
print("-" * 35)
for s_val, p_val in [(1, 0), (1, 1), (2, 0), (2, 1)]:
    size = conv_output_size(M_f, K_f, s_val, p_val)
    same = " ← same padding" if size == M_f else ""
    print(f"{s_val:>8} | {p_val:>8} | {size:>8}{same}")

# =============================================================================
# 5. PRACTICE QUESTIONS
# =============================================================================

print("\n" + "="*60)
print("5. PRACTICE QUESTIONS")
print("="*60)

# Q1: Zero kernel → all-zero output
print("\nQ1: Kernel of zeros (3×3) applied to random 4×4 image")
torch.manual_seed(42)
img_q1 = torch.randn((1, 1, 4, 4))
conv_q1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv_q1.state_dict()['weight'][0][0] = torch.zeros(3, 3)
conv_q1.state_dict()['bias'][0] = 0.0
z_q1 = conv_q1(img_q1)
print(f"Output: {z_q1.detach()}")
print(f"✓ All zeros — element-wise product with 0 kernel = 0 for every region.")
print(f"  Output shape: {z_q1.shape[2:]} (M=4, K=3 → 4-3+1=2)")

# Q2: Output size M=4, K=2, s=2
print("\nQ2: Output size for M=4, K=2, stride=2")
M_q2, K_q2, s_q2 = 4, 2, 2
size_q2 = conv_output_size(M_q2, K_q2, s_q2)
print(f"floor(({M_q2}-{K_q2})/{s_q2}) + 1 = {size_q2}")
print(f"✓ Output size = {size_q2} × {size_q2}")

print("\n" + "="*60)
print("Summary Table: conv_output_size(M, K, s, p)")
print("="*60)
test_cases = [
    (5, 3, 1, 0, "basic, no padding"),
    (4, 2, 1, 0, "basic, no padding"),
    (4, 2, 2, 0, "stride=2"),
    (4, 2, 3, 1, "stride=3, padding=1"),
    (6, 3, 1, 1, "same padding"),
]
print(f"{'M':>4} {'K':>4} {'s':>4} {'p':>4} | {'Output':>8} | Note")
print("-" * 55)
for M_, K_, s_, p_, note in test_cases:
    out = conv_output_size(M_, K_, s_, p_)
    print(f"{M_:>4} {K_:>4} {s_:>4} {p_:>4} | {out:>8} | {note}")
