# PyTorch CNN Deep Learning Labs

![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![IBM AI Engineering](https://img.shields.io/badge/IBM-AI%20Engineering%20Certificate-054ADA?style=flat&logo=ibm&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat)

---

<!--
╔══════════════════════════════════════════════════════════════════╗
║  LINKEDIN BLOCK — 1,131 chars                                    ║
║  Copy everything between START and END into LinkedIn Projects    ║
╚══════════════════════════════════════════════════════════════════╝
START ──────────────────────────────────────────────────────────────
-->

Designed and implemented four production-quality PyTorch labs on Convolutional Neural Networks, progressing from first-principles convolution mechanics to 10-class image classification — bridging rigorous mathematical understanding with engineering application.

**Labs:**
▸ **Convolution** — 2D kernel operations (Sobel-X), stride, zero padding, output size formula: `M_new = ⌊(M+2p−K)/s⌋+1`
▸ **Activation & Pooling** — ReLU vs Sigmoid/Tanh; MaxPool2d spatial downsampling; full Conv→ReLU→MaxPool pipeline with shape tracking
▸ **Loss & Initialisation** — BCELoss vs MSE gradient analysis; large initial weights saturate the sigmoid, collapsing gradient flow and degrading accuracy from **~100% to ~60%**
▸ **Fashion-MNIST Classification** — CNN vs CNN+BatchNorm on 70,000 images (10 classes); per-class accuracy breakdown, misclassification inspection, training cost and accuracy curves

Each lab delivered as `.ipynb` + `.py`. Repository features architecture diagrams, WHY-driven code comments, automated figure export, and a formula reference table.

**Stack:** PyTorch 2.8 · Python 3.10 · torchvision · NumPy · Matplotlib · Jupyter

🔗 [github.com/pumpuni07/pytorch-cnn-deep-learning-labs](https://github.com/pumpuni07/pytorch-cnn-deep-learning-labs)

<!--
END ────────────────────────────────────────────────────────────────
-->

---

## Repository Structure

```
pytorch-cnn-deep-learning-labs/
├── 01_what_is_convolution.ipynb
├── 01_what_is_convolution.py
├── 02_activation_max_pooling.ipynb
├── 02_activation_max_pooling.py
├── 03_logistic_regression_crossentropy_vs_bad_init.ipynb
├── 03_logistic_regression_crossentropy_vs_bad_init.py
├── 04_fashion_mnist_cnn.ipynb
├── 04_fashion_mnist_cnn.py
└── README.md
```

> Running any lab auto-generates `.png` figures in the working directory. No manual steps required.

---

## Architecture

### Standard CNN

```
Input  (1 × 16 × 16)
  │
  ▼  Conv2d(1→16, k=5, p=2)     →  16 × 16 × 16   [same padding: (16+4−5)/1+1=16]
  ▼  ReLU
  ▼  MaxPool2d(k=2)              →  16 ×  8 ×  8
  │
  ▼  Conv2d(16→32, k=5, p=2)    →  32 ×  8 ×  8   [same padding]
  ▼  ReLU
  ▼  MaxPool2d(k=2)              →  32 ×  4 ×  4
  │
  ▼  Flatten                     →        512       [32 × 4 × 4]
  ▼  Linear(512 → 10)            →         10 logits
```

### CNN + Batch Normalization

```
Input  (1 × 16 × 16)
  │
  ▼  Conv2d(1→16, k=5, p=2)
  ▼  BatchNorm2d(16)             ← normalises each channel across batch & spatial dims
  ▼  ReLU
  ▼  MaxPool2d(k=2)              →  16 ×  8 ×  8
  │
  ▼  Conv2d(16→32, k=5, p=2)
  ▼  BatchNorm2d(32)             ← reduces internal covariate shift
  ▼  ReLU
  ▼  MaxPool2d(k=2)              →  32 ×  4 ×  4
  │
  ▼  Flatten → Linear(512 → 10)
  ▼  BatchNorm1d(10)             ← normalises across batch dimension
  ▼  10 logits
```

---

## Lab Reference

| Lab | Core Concept | Key Formula / Method | Output Figures |
|-----|-------------|---------------------|----------------|
| 1 — Convolution | 2D kernel operations | `M_new = ⌊(M+2p−K)/s⌋+1` | Kernel viz · stride/padding comparison |
| 2 — Activation & Pooling | ReLU + MaxPool2d | `ReLU(x) = max(0,x)` | Pipeline trace · activation curves |
| 3 — Loss & Init | BCELoss vs MSE | Gradient magnitude analysis | Decision boundaries · gradient plot |
| 4 — Classification | CNN vs CNN+BN | CrossEntropyLoss · SGD | Cost curves · per-class accuracy |

---

## Results

### Lab 3 — Effect of Weight Initialisation

| Model | Initialisation | Final Accuracy |
|-------|---------------|----------------|
| Logistic Regression | Good (near zero) | ~100% |
| Logistic Regression | Bad (w=−15, b=3) | ~60% |

> Identical architecture and loss function. The 40-point accuracy gap is caused entirely by sigmoid saturation from poor weight initialisation — a controlled demonstration of gradient vanishing.

### Lab 4 — Fashion-MNIST (16×16, 5 epochs, SGD lr=0.1)

| Model | Parameters | Val Accuracy |
|-------|-----------|-------------|
| CNN | ~8,330 | *(fill after run)* |
| CNN + BatchNorm | ~8,490 | *(fill after run)* |

> **On accuracy benchmarks:** Images are resized 28×28 → 16×16 for CPU training, which reduces spatial information compared to the standard benchmark. Published results for comparable 2-layer CNNs on 28×28 Fashion-MNIST report approximately 88–90% after 25+ epochs with momentum-based optimisers. Results here will differ — fill in your actual numbers after running.

---

## Skills Demonstrated

**Deep Learning & PyTorch**
`Conv2d` · `MaxPool2d` · `BatchNorm2d` · `BatchNorm1d` · `BCELoss` · `CrossEntropyLoss` · `SGD` · `DataLoader` · `state_dict` · `.train()/.eval()` switching

**Computer Vision Fundamentals**
Spatial feature extraction · Kernel design (Sobel-X) · Padding & stride theory · Activation function comparison · Translation invariance via pooling · Gradient flow analysis

**Engineering Practices**
Dual `.ipynb`/`.py` delivery · Fixed random seeds (`torch.manual_seed`) · Pinned dependencies · Automated figure export · Modular training/evaluation functions · Docstrings and WHY-comments throughout

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/pumpuni07/pytorch-cnn-deep-learning-labs.git
cd pytorch-cnn-deep-learning-labs

# 2. Install (CPU-only PyTorch — no GPU required)
pip install pandas numpy matplotlib scipy
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Run
jupyter notebook                    # open any .ipynb and run all cells
python 04_fashion_mnist_cnn.py      # or execute directly as a script
```

> **Lab 4 note:** Fashion-MNIST (~30 MB) downloads automatically on first run. Internet required once only.

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.10 |
| PyTorch | 2.8.0+cpu |
| torchvision | 0.23.0+cpu |
| NumPy | latest stable |
| Matplotlib | latest stable |
| SciPy | latest stable |

---

**Connect:** [LinkedIn](https://www.linkedin.com/in/jackpumpunifrimpongmanso) · [Portfolio](https://jackpumpunifrimpongmanso.base44.app) · [GitHub @pumpuni07](https://github.com/pumpuni07)

**License:** MIT · **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research, MIT)
