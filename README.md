# CIFAR-10 Image Classification — PyTorch CNN

A custom Convolutional Neural Network (CNN) built from scratch in PyTorch to classify images from the CIFAR-10 dataset across 10 categories (planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks).

## Result

**Test Accuracy: 86.22%**
---

## Architecture

4 convolutional blocks with progressively increasing filter depth (32 → 64 → 128 → 256), followed by a fully connected classifier.

```
Input (3×32×32)
  → Conv Block 1: Conv2d(3, 32)  → ReLU → MaxPool → BatchNorm
  → Conv Block 2: Conv2d(32, 64) → ReLU → MaxPool → BatchNorm
  → Conv Block 3: Conv2d(64,128) → ReLU → MaxPool → BatchNorm
  → Conv Block 4: Conv2d(128,256)→ ReLU → MaxPool → BatchNorm
  → Flatten
  → Dropout(0.3)
  → Linear(1024, 256) → ReLU
  → Linear(256, 10)
Output: 10-class softmax
```

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| Batch normalisation after each pooling layer | Stabilises training, allows higher learning rates |
| Dropout (0.3) before classifier | Reduces overfitting on training set |
| Data augmentation (crops, flips, colour jitter) | Applied to training set only to improve generalisation |
| Train/val split with fixed seed (42) | Ensures reproducible evaluation |
| Adam optimiser, lr=0.001 | Adaptive learning rate, fast convergence |
| CrossEntropyLoss | Standard for multi-class classification |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Optimiser | Adam (lr=0.001) |
| Train/Val split | 80/20 (40,000/10,000) |
| Device | CPU |

---

## How to Run

**Requirements**
```bash
pip install torch torchvision
```

**Train the model**
```bash
python train.py
```
This will download the CIFAR-10 dataset automatically, train for 50 epochs, save the model weights to `IMLO_Coursework.pth`, and print test accuracy.

**Test a saved model**
```bash
python test.py
```

---

## Technologies

- Python
- PyTorch
- torchvision
