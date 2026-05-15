# 🧠 CNN from Scratch — Fashion MNIST
### UE24CS645BC2 | Deep Learning Theory and Practice | Assignment 1

> A complete Convolutional Neural Network built **entirely from first principles using only NumPy** — no PyTorch, no TensorFlow, no Keras (only used to load the dataset).

---

## 📌 What This Project Does

This project answers the question: **"How does a CNN actually work under the hood?"**

Every single component is hand-coded from scratch:
- The math behind convolution
- How gradients flow backwards through each layer
- How the network learns from its mistakes

The model is trained and evaluated on **Fashion MNIST** — 70,000 grayscale images of clothing across 10 categories.

---

## 📁 Project Structure

```
UE24CS645BC2__Fashion_MNIST_CNN/
│
├── Fashion_MNIST_CNN_Colab.ipynb   ← Main notebook (run this)
├── cnn_from_scratch.py             ← Standalone Python script
└── README.md                       ← This file
```

---

## 🏗️ CNN Architecture

```
Input Image (28×28 grayscale)
        │
        ▼
┌─────────────────────────────┐
│  Conv2D (8 filters, 3×3)    │  ← Detects basic edges & textures
│  + ReLU                     │
│  + MaxPool (2×2)            │  → Output: 8 × 14 × 14
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Conv2D (16 filters, 3×3)   │  ← Detects complex patterns
│  + ReLU                     │
│  + MaxPool (2×2)            │  → Output: 16 × 7 × 7
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Flatten                    │  → Output: 784 values
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Fully Connected (128)      │
│  + ReLU                     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Fully Connected (10)       │  ← One output per class
│  + Softmax                  │
└─────────────────────────────┘
        │
        ▼
  Class Prediction 🎽
```

---

## 🔧 Layers Implemented from Scratch

### 1. Convolutional Layer
The core of the CNN. A small filter (kernel) slides across the image and computes a dot product at every position.

**Forward pass:**
```
out[n, f, i, j] = Σ (W[f] · patch(X, i, j)) + b[f]
```

**Backward pass:**
- `dW` = how much each filter weight contributed to the error
- `db` = how much each bias contributed
- `dX` = gradient passed back to the previous layer

### 2. ReLU Activation
Kills negative values. Keeps the network non-linear.
```
f(x) = max(0, x)
Backward: gradient passes through where x > 0, else 0
```

### 3. MaxPooling Layer
Shrinks the feature map by keeping only the **maximum value** in each window. Makes the network robust to small shifts in the image.

**Backward:** Gradient only flows back to whichever position had the max value.

### 4. Flatten Layer
Converts the 3D feature map `(channels × height × width)` into a 1D vector so it can be fed into the fully connected layer.

### 5. Fully Connected Layer
Every neuron connects to every input. Standard matrix multiplication:
```
out = X @ W + b
```

### 6. Softmax + Cross-Entropy Loss
Converts raw scores (logits) into probabilities. The combined gradient is:
```
∂L/∂logits = (softmax(logits) − one_hot(y)) / N
```
This is numerically stable and avoids computing the Jacobian of softmax separately.

---

## ⚙️ How Training Works

```
For each epoch:
  Shuffle the training data
  For each mini-batch:
    1. Forward Pass  → compute predictions
    2. Compute Loss  → how wrong are we?
    3. Backward Pass → compute gradients (backpropagation)
    4. Update Params → optimizer adjusts weights
  Evaluate on test set → report accuracy
```

**Optimizer:** SGD with Momentum
```
velocity = momentum × velocity − lr × gradient
weights  = weights + velocity
```
Momentum (μ=0.9) smooths out noisy gradients and speeds up convergence.

---

## 📊 Dataset: Fashion MNIST

| Property | Value |
|----------|-------|
| Total images | 70,000 |
| Training set | 60,000 |
| Test set | 10,000 |
| Image size | 28 × 28 pixels |
| Color | Grayscale |
| Classes | 10 |

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `Fashion_MNIST_CNN_Colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU** (for faster training)
4. **Runtime → Run all**

The notebook will:
- Load Fashion MNIST automatically via Keras
- Train the CNN for 10 epochs
- Plot training curves, per-class accuracy, and sample predictions

### Option 2 — Run Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/UE24CS645BC2__Fashion_MNIST_CNN.git
cd UE24CS645BC2__Fashion_MNIST_CNN

# Install dependencies
pip install numpy tensorflow matplotlib

# Run
python cnn_from_scratch.py
```

---

## 📈 Expected Results

| Setting | Train Accuracy | Test Accuracy |
|---------|---------------|---------------|
| 10k subset, 10 epochs | ~78% | ~73–75% |
| Full 60k, 10 epochs | ~85% | ~80–83% |

Training time:
- **CPU:** ~15–20 min (10k subset), ~2–3 hours (full)
- **GPU (Colab T4):** ~3–5 min (10k subset), ~25–35 min (full)

---

## 📦 Dependencies

| Library | Used For |
|---------|----------|
| `numpy` | All CNN math — convolution, backprop, matrix ops |
| `tensorflow.keras` | **Only** to download the Fashion MNIST dataset |
| `matplotlib` | Plotting training curves and predictions |

> ⚠️ Keras is used **only as a data loader**. The CNN itself uses zero ML framework code.

---

## 🧪 Notebook Walkthrough

| Step | What Happens |
|------|-------------|
| Step 1 | Import libraries |
| Step 2 | Load Fashion MNIST via Keras |
| Step 3 | Visualize one sample per class |
| Step 4 | Define all CNN layers from scratch |
| Step 5 | Assemble full CNN model |
| Step 6 | Define SGD with Momentum optimizer |
| Step 7 | Define training & evaluation functions |
| Step 8 | **Train the model** |
| Step 9 | Plot loss & accuracy curves |
| Step 10 | Per-class accuracy table |
| Step 11 | Per-class accuracy bar chart |
| Step 12 | Visualize 20 predictions (green=correct, red=wrong) |
| Step 13 | Visualize learned conv filters |
| Step 14 | Save results to `results.json` |

