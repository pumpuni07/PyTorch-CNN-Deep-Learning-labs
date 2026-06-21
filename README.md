# PyTorch CNN Deep Learning Labs

![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![IBM AI Engineering](https://img.shields.io/badge/IBM-AI%20Engineering%20Professional%20Certificate-054ADA?style=flat&logo=ibm&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> **Production-quality PyTorch implementations of core CNN concepts** — from first principles of convolution through to multi-class image classification with Batch Normalization. Each lab ships as both a fully documented Jupyter Notebook (`.ipynb`) and a clean Python script (`.py`).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Lab Summaries](#lab-summaries)
  - [Lab 1 — What is Convolution?](#lab-1--what-is-convolution)
  - [Lab 2 — Activation Functions & Max Pooling](#lab-2--activation-functions--max-pooling)
  - [Lab 3 — Logistic Regression: Cross Entropy vs Bad Initialization](#lab-3--logistic-regression-cross-entropy-vs-bad-initialization)
  - [Lab 4 — Fashion-MNIST: CNN vs CNN + Batch Normalization](#lab-4--fashion-mnist-cnn-vs-cnn--batch-normalization)
- [CNN Architecture Diagrams](#cnn-architecture-diagrams)
- [Results](#results)
- [Skills Demonstrated](#skills-demonstrated)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [LinkedIn](#linkedin)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This repository contains four end-to-end PyTorch labs covering the foundational building blocks of Convolutional Neural Networks (CNNs). The series progresses deliberately — each lab adds one layer of complexity to the previous, building toward a full image classification pipeline on Fashion-MNIST.

**Who this is for:**
- ML Engineers and practitioners wanting clear, well-documented PyTorch CNN reference code
- Reviewers of the IBM AI Engineering Professional Certificate curriculum
- Anyone learning how convolution, pooling, activation functions, and batch normalization connect in practice

**What makes this different from generic tutorials:**
- Every lab includes both `.ipynb` and `.py` — runnable end-to-end in either environment
- All figures are saved to disk automatically (no manual screenshot required)
- Architecture decisions are explained in comments alongside the code, not just described
- Accuracy comparisons are honest about setup conditions (image size, epochs, optimizer)

**Course:** IBM AI Engineering Professional Certificate — Deep Neural Networks with PyTorch

---

## Repository Structure

```
pytorch-cnn-deep-learning-labs/
│
├── 01_what_is_convolution.ipynb               # Lab 1 — Jupyter Notebook
├── 01_what_is_convolution.py                  # Lab 1 — Python script
│
├── 02_activation_max_pooling.ipynb            # Lab 2 — Jupyter Notebook
├── 02_activation_max_pooling.py               # Lab 2 — Python script
│
├── 03_logistic_regression_crossentropy_       # Lab 3 — Jupyter Notebook
│   vs_bad_init.ipynb
├── 03_logistic_regression_crossentropy_       # Lab 3 — Python script
│   vs_bad_init.py
│
├── 04_fashion_mnist_cnn.ipynb                 # Lab 4 — Jupyter Notebook
├── 04_fashion_mnist_cnn.py                    # Lab 4 — Python script
│
└── README.md
```

> **Note:** Running any lab will auto-generate `.png` figures in the same directory (e.g. `convolution_demo.png`, `fashion_mnist_results.png`). No manual steps needed.

---

## Lab Summaries

---

### Lab 1 — What is Convolution?

<details>
<summary><strong>Click to expand</strong></summary>

#### Concept

Convolution is a linear operation that slides a kernel (filter) across an input image, computing element-wise products and summing them at each position. Unlike a fully-connected layer, the kernel **shares weights** across all spatial positions — this gives CNNs their parameter efficiency and spatial awareness.

Mathematical progression from scalar to tensor:

| Operation | Formula |
|---|---|
| Linear (scalar) | $y = wx + b$ |
| Linear (vector) | $\mathbf{y} = \mathbf{w}\mathbf{x} + b$ |
| Matrix multiplication | $\mathbf{y} = \mathbf{w}\mathbf{X} + \mathbf{b}$ |
| **Convolution** | $\mathbf{Y} = \mathbf{w} * \mathbf{X} + \mathbf{b}$ |

#### Output Size Formula

| Configuration | Formula |
|---|---|
| Basic (no stride/padding) | $M_{new} = M - K + 1$ |
| With stride $s$ and padding $p$ | $M_{new} = \lfloor(M + 2p - K) / s\rfloor + 1$ |
| Same padding (stride=1) | $p = (K - 1) / 2$ |

#### Key Code Snippet

```python
import torch
import torch.nn as nn

# Sobel-X kernel — detects vertical edges
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv.state_dict()['weight'][0][0] = torch.tensor([[ 1.0,  0.0, -1.0],
                                                    [ 2.0,  0.0, -2.0],
                                                    [ 1.0,  0.0, -1.0]])
conv.state_dict()['bias'][0] = 0.0

# Input: (N=1, C=1, H=5, W=5)
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1  # vertical edge at column 2

output = conv(image)
# Output shape: (1, 1, 3, 3)  → M_new = 5 - 3 + 1 = 3
```

#### Output Figures
- `convolution_demo.png` — Input image, Sobel-X kernel, and output feature map side by side
- `output_size_vs_kernel.png` — Bar chart of output size as kernel size increases
- `stride_padding_comparison.png` — How stride and padding interact with output dimensions

</details>

---

### Lab 2 — Activation Functions & Max Pooling

<details>
<summary><strong>Click to expand</strong></summary>

#### Concept

**Why activation functions?** Convolution is linear. Stacking linear operations always collapses to a single linear operation regardless of depth. Activation functions introduce the non-linearity that allows deep networks to approximate complex functions.

**ReLU** (Rectified Linear Unit) is the standard choice in CNNs:

$$\text{ReLU}(x) = \max(0, x)$$

- Avoids vanishing gradients (unlike sigmoid/tanh for deep networks)
- Fast to compute
- Applied element-wise to every value in the activation map

**Max Pooling** performs spatial downsampling by retaining only the maximum value in each pooling window:
- Reduces spatial dimensions → fewer parameters in subsequent layers
- Provides local translation invariance
- Output size follows the same formula as convolution: $\lfloor(M - K) / s\rfloor + 1$

#### CNN Building Block

```
Input → Conv2d → ReLU → MaxPool2d → (next layer)
```

#### Key Code Snippet

```python
# Full pipeline: Conv → ReLU → MaxPool
conv_layer = nn.Conv2d(1, 1, kernel_size=3, padding=0)
relu_layer = nn.ReLU()
pool_layer = nn.MaxPool2d(2, stride=2)

# Shape tracking (7×7 input)
Z = conv_layer(image)   # 7 → 5  (7 - 3 + 1 = 5)
A = relu_layer(Z)       # 5 → 5  (unchanged)
P = pool_layer(A)       # 5 → 2  (floor((5-2)/2) + 1 = 2)
```

#### Output Figures
- `relu_activation.png` — Activation map before and after ReLU
- `activation_functions_comparison.png` — ReLU vs Sigmoid vs Tanh curves
- `max_pooling_demo.png` — MaxPool with stride=1 vs stride=2 on the same input
- `cnn_pipeline_demo.png` — Four-stage visual: Input → Conv → ReLU → MaxPool

</details>

---

### Lab 3 — Logistic Regression: Cross Entropy vs Bad Initialization

<details>
<summary><strong>Click to expand</strong></summary>

#### Concept

**Why Cross Entropy (BCE) instead of MSE for classification?**

MSE on sigmoid outputs produces extremely flat gradients when predictions are far from the true label — the sigmoid saturates near 0 or 1, and MSE gradient $\partial L / \partial p = 2(p - y)$ becomes near zero. This is gradient saturation.

Binary Cross Entropy (BCE) is derived from Maximum Likelihood Estimation:

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

BCE gradient $\partial L / \partial p = -(y/p) + (1-y)/(1-p)$ provides large gradients when predictions are wrong — the optimizer receives a strong training signal throughout.

**Bad Initialization** forces large initial weights, which saturates the sigmoid at epoch 1. Near-zero gradients from the start mean the model barely updates over 100 epochs.

| | Good Init | Bad Init |
|---|---|---|
| Initial weights | Small (near zero) | Large (e.g. w=−15) |
| Sigmoid output at start | ~0.5 (unsaturated) | ~0 or ~1 (saturated) |
| Gradient magnitude | Healthy | Near zero |
| Final accuracy | ~100% | ~60% |

#### Key Code Snippet

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2.0)

# Bad initialization (manual override)
model.state_dict()['linear.weight'][0] = torch.tensor([-15.0])
model.state_dict()['linear.bias'][0]   = torch.tensor([3.0])
```

#### Output Figures
- `binary_dataset.png` — Dataset scatter plot
- `good_vs_bad_init.png` — Loss curves and sigmoid decision boundaries side by side
- `bce_vs_mse_gradients.png` — Gradient magnitude comparison: BCE vs MSE

</details>

---

### Lab 4 — Fashion-MNIST: CNN vs CNN + Batch Normalization

<details>
<summary><strong>Click to expand</strong></summary>

#### Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research) — 70,000 grayscale images, 10 clothing categories, 28×28 pixels.

> **Note:** In this lab, images are resized to **16×16** for faster CPU training. This reduces spatial information relative to the standard 28×28 benchmark and should be accounted for when comparing accuracy to published results.

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

#### Model Comparison

Both models share the same two-block backbone. `CNN_batch` adds Batch Normalization after each convolutional layer and the final FC layer.

**What Batch Normalization does:**
- Normalises layer inputs to zero mean and unit variance across the batch
- Reduces internal covariate shift — activations stay in a healthy range throughout training
- Acts as a mild regulariser (reduces sensitivity to weight initialisation)
- Often enables faster convergence

#### Key Code Snippet

```python
class CNN_batch(nn.Module):
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super().__init__()
        self.cnn1      = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.conv1_bn  = nn.BatchNorm2d(out_1)   # ← BatchNorm after Conv
        self.maxpool1  = nn.MaxPool2d(kernel_size=2)
        self.cnn2      = nn.Conv2d(out_1, out_2, kernel_size=5, padding=2)
        self.conv2_bn  = nn.BatchNorm2d(out_2)   # ← BatchNorm after Conv
        self.maxpool2  = nn.MaxPool2d(kernel_size=2)
        self.fc1       = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1    = nn.BatchNorm1d(number_of_classes)  # ← BatchNorm after FC

    def forward(self, x):
        x = torch.relu(self.conv1_bn(self.cnn1(x)))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2_bn(self.cnn2(x)))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        return self.bn_fc1(self.fc1(x))
```

#### Output Figures
- `fashion_mnist_samples.png` — Sample grid of validation images
- `fashion_mnist_results.png` — Training cost and validation accuracy curves for both models
- `fashion_mnist_per_class.png` — Per-class accuracy bar chart: CNN vs CNN+BatchNorm
- `fashion_mnist_errors.png` — Misclassified samples with true vs predicted labels

</details>

---

## CNN Architecture Diagrams

### Standard CNN

```
Input (1 × 16 × 16)
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv2d(1→16, kernel=5, padding=2)      │  → 16 × 16 × 16  [same padding]
│  ReLU                                   │
│  MaxPool2d(kernel=2)                    │  → 16 × 8  × 8
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv2d(16→32, kernel=5, padding=2)     │  → 32 × 8  × 8   [same padding]
│  ReLU                                   │
│  MaxPool2d(kernel=2)                    │  → 32 × 4  × 4
└─────────────────────────────────────────┘
        │
        ▼
   Flatten → 32 × 4 × 4 = 512
        │
        ▼
   Linear(512 → 10)
        │
        ▼
   Output logits (10 classes)
```

### CNN + Batch Normalization

```
Input (1 × 16 × 16)
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv2d(1→16, kernel=5, padding=2)      │  → 16 × 16 × 16
│  BatchNorm2d(16)          ← NEW         │  normalises across batch & spatial
│  ReLU                                   │
│  MaxPool2d(kernel=2)                    │  → 16 × 8  × 8
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv2d(16→32, kernel=5, padding=2)     │  → 32 × 8  × 8
│  BatchNorm2d(32)          ← NEW         │  normalises across batch & spatial
│  ReLU                                   │
│  MaxPool2d(kernel=2)                    │  → 32 × 4  × 4
└─────────────────────────────────────────┘
        │
        ▼
   Flatten → 512
        │
        ▼
   Linear(512 → 10)
   BatchNorm1d(10)           ← NEW        normalises across batch
        │
        ▼
   Output logits (10 classes)
```

---

## Results

### Lab 3 — Logistic Regression

| Model | Initialization | Final Accuracy |
|---|---|---|
| Logistic Regression (BCE) | Good (near zero) | ~100% |
| Logistic Regression (BCE) | Bad (w=−15, b=3) | ~60% |

> **Key insight:** Both models use identical architecture and loss function. The accuracy gap is caused entirely by gradient saturation from poor weight initialization.

### Lab 4 — Fashion-MNIST

| Model | Parameters | Val Accuracy (5 epochs) |
|---|---|---|
| CNN | ~8,330 | [Run to obtain] |
| CNN + BatchNorm | ~8,490 | [Run to obtain] |

> **Important note on accuracy:** Images in this lab are resized to **16×16** (vs. the standard 28×28 benchmark). Published benchmarks for simple 2-layer CNNs on 28×28 Fashion-MNIST report approximately 88–90% after 25+ epochs with momentum-based optimizers. Results in this lab will differ due to the smaller image size and fewer training epochs. Run the notebooks on your hardware to obtain your actual numbers and fill them in above.
>
> For reference: <cite index="1-1">a comparable PyTorch BN-CNN trained on the standard 28×28 images achieved approximately 89.9% validation accuracy after 5 epochs.</cite>

---

## Skills Demonstrated

**Deep Learning & Neural Networks**
- Building and training CNNs in PyTorch from scratch using `nn.Module`
- Understanding and applying `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `BatchNorm1d`, `ReLU`
- Implementing multi-class classification with `CrossEntropyLoss` and binary classification with `BCELoss`
- Weight initialization effects and gradient saturation diagnosis

**PyTorch Proficiency**
- Custom `Dataset` and `DataLoader` construction
- `state_dict` manipulation for kernel assignment
- Model training loops with `.train()` / `.eval()` mode switching
- Gradient flow management with `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`
- Shape tracking and tensor reshaping with `.view()`

**Computer Vision Fundamentals**
- 2D convolution mechanics: kernel, stride, padding, output size calculation
- Spatial downsampling via max pooling vs stride
- Feature map visualisation with Matplotlib
- Fashion-MNIST data pipeline with `torchvision.transforms`

**Software Engineering Practices**
- Dual-format delivery: `.ipynb` for exploration, `.py` for production
- Docstrings and inline comments throughout
- Modular training and evaluation functions with clean separation of concerns
- Automated figure export with `plt.savefig()`

---

## Quick Start

### Prerequisites

Python 3.10+ is required. A CPU-only environment is fully supported.

### 1. Clone the repository

```bash
git clone https://github.com/pumpuni07/pytorch-cnn-deep-learning-labs.git
cd pytorch-cnn-deep-learning-labs
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scipy
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3a. Run as Jupyter Notebooks

```bash
pip install jupyter
jupyter notebook
```

Then open any `.ipynb` file and run all cells (`Kernel → Restart & Run All`).

### 3b. Run as Python scripts

```bash
# Lab 1 — Convolution fundamentals
python 01_what_is_convolution.py

# Lab 2 — Activation functions and max pooling
python 02_activation_max_pooling.py

# Lab 3 — Logistic regression: cross entropy vs bad initialization
python 03_logistic_regression_crossentropy_vs_bad_init.py

# Lab 4 — Fashion-MNIST CNN vs CNN+BatchNorm (downloads dataset on first run)
python 04_fashion_mnist_cnn.py
```

> **Lab 4 note:** The Fashion-MNIST dataset (~30 MB) is downloaded automatically on first run to `.fashion/data/`. An internet connection is required for the first execution only.

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | 2.8.0+cpu |
| torchvision | 0.23.0+cpu |
| torchaudio | 2.8.0+cpu |
| numpy | latest stable |
| matplotlib | latest stable |
| scipy | latest stable |
| Pillow | latest stable (via torchvision) |

> All version-pinned installs use the official PyTorch CPU wheel index: `https://download.pytorch.org/whl/cpu`

---

## LinkedIn

These labs are part of an active professional portfolio.
Connect on LinkedIn — [Jack Pumpuni Frimpong-Manso](https://www.linkedin.com/in/jackpumpunifrimpongmanso) — for more projects spanning RAG pipelines, geospatial AI, and climate-informed machine learning.

---

## Acknowledgements

- **IBM** — AI Engineering Professional Certificate curriculum and lab design
- **Zalando Research** — [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) (MIT License)
- **PyTorch** — Open-source deep learning framework ([pytorch.org](https://pytorch.org))
- **Joseph Santarcangelo (IBM)** — Original lab author

---

## License

This repository is released under the [MIT License](LICENSE).

The Fashion-MNIST dataset is copyright Zalando Research and distributed under its own [MIT License](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
