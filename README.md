# Lightweight Multi-Layer Perceptron (MLP) from Scratch in C

[![Language: C](https://img.shields.io/badge/Language-C-00599C.svg?style=flat-square&logo=c)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Dependencies: None](https://img.shields.io/badge/Dependencies-None-brightgreen.svg?style=flat-square)](https://en.wikipedia.org/wiki/Standard_C_library)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A highly optimized, educational, and memory-safe Multi-Layer Perceptron (MLP) implemented entirely from scratch in pure C. This codebase has **no external math or ML dependencies** (no BLAS, no PyTorch, no GSL) and is designed for maximum performance, hardware cache friendliness, and strict heap-safety.

---

## 🚀 Key Technical Achievements

### 1. Zero Runtime Allocations
Traditional C neural network implementations allocate and free matrices inside the training and forward/backward propagation loops. This causes severe heap fragmentation and significant `malloc`/`free` overhead. 
In this project, **all activation vectors, pre-activation state matrices (`z`), and error gradients (`d_weights`, `d_biases`, `d_activations`, `d_zs`) are pre-allocated** during network initialization. The training loop executes with **zero dynamic memory operations**, offering deterministic execution speed and guaranteed protection against leaks.

### 2. Cache-Friendly flat Matrix Layouts
Rather than using traditional nested pointers (`double**`), which result in scattered non-contiguous memory allocations and high cache miss rates, this repository structures matrices as contiguous 1D arrays:
```c
typedef struct {
    int rows;
    int cols;
    double *data; // Contiguous block of rows * cols doubles
} Matrix;
```
This enables consecutive element lookups to hit the CPU L1/L2 caches, resulting in faster math operations.

### 3. Integrated GEMM (General Matrix Multiply)
To support forward and backward propagation without copying or transposing matrices in memory, we implemented a custom, lightweight version of the standard BLAS **GEMM** routine:
$$\mathbf{C} \leftarrow \alpha \cdot \text{op}(\mathbf{A}) \cdot \text{op}(\mathbf{B}) + \beta \cdot \mathbf{C}$$
where $\text{op}(\mathbf{X})$ is either $\mathbf{X}$ or the transpose $\mathbf{X}^T$. This single function elegantly handles:
- **Forward weighted inputs**: $\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$
- **Weight gradients calculation**: $\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$
- **Previous activation gradients backprop**: $\frac{\partial L}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \boldsymbol{\delta}^{[l]}$

---

## 🧮 Mathematical Foundations & Calculus

### 1. Activation Functions & Derivatives
The network supports two activation functions:
- **ReLU** (Hidden Layers):
  $$g(z) = \max(0, z)$$
  $$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$$
- **Sigmoid** (Output Layer):
  $$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  $$g'(z) = a(1 - a) \quad \text{where } a = \sigma(z)$$

### 2. Parameter Initialization
Poor weight initialization leads to vanishing or exploding gradients. We implement:
- **He Initialization** (for ReLU layers): Weights are drawn from a uniform distribution:
  $$W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{N_{\text{in}}}}, \sqrt{\frac{6}{N_{\text{in}}}}\right)$$
- **Xavier Initialization** (for Sigmoid layers): Weights are drawn from:
  $$W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{N_{\text{in}} + N_{\text{out}}}}, \sqrt{\frac{6}{N_{\text{in}} + N_{\text{out}}}}\right)$$

### 3. Backpropagation Equations
For Mean Squared Error (MSE) loss on a single sample:
$$L = \frac{1}{2} (\mathbf{a}^{[L]} - \mathbf{y})^2$$

Working backwards through each layer $l$:
1. **Output Layer Error Gradient**:
   $$\frac{\partial L}{\partial \mathbf{a}^{[L]}} = \mathbf{a}^{[L]} - \mathbf{y}$$
2. **Delta (Gradients w.r.t pre-activation $\mathbf{z}$)**:
   $$\boldsymbol{\delta}^{[l]} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \odot g'(\mathbf{z}^{[l]})$$
   *(where $\odot$ denotes the element-wise Hadamard product)*
3. **Weight Gradients**:
   $$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$$
4. **Bias Gradients**:
   $$\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$$
5. **Backpropagation of Activations (to layer $l-1$)**:
   $$\frac{\partial L}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \boldsymbol{\delta}^{[l]}$$

---

## 📂 Project Architecture

```
├── matrix.h   # Core matrix struct and inline-transposed math prototypes
├── matrix.c   # Contiguous memory allocators, element-wise ops, and GEMM
├── nn.h       # Layer & Network structures, activation enums & derivatives
├── nn.c       # Weights initialization, forward pass, and backpropagation loop
├── main.c     # Test driver solving the classic XOR logic gate
├── Makefile   # Cross-platform compiler settings (GCC/Clang, -O3 optimizations)
└── README.md  # Portfolio documentation
```

---

## 🛠️ Build and Execution

### Prerequisites
Make sure you have GCC (or Clang) and GNU Make installed on your system.

### Compiling
To compile the project with `-O3` speed optimizations, run:
```bash
make
```

### Running the XOR Verification
Execute the compiled binary to train the model and see final predictions:
```bash
./nn_xor
```

### Cleaning Artifacts
To delete object files and executable binaries:
```bash
make clean
```

---

## 📊 Verification Metrics (XOR Gate Problem)

Because XOR is non-linearly separable, a simple linear model fails to solve it. The 3-layer model successfully constructs a non-linear decision boundary:

```
=== Training Multi-Layer Perceptron from Scratch in C ===

Configuring MLP architecture:
  Input layer size: 2
  Hidden layer size: 4 (Activation: ReLU)
  Output layer size: 1 (Activation: Sigmoid)

Starting training: epochs = 15000, learning_rate = 0.15...
  Epoch     1/15000 - Average MSE Loss: 0.132402
  Epoch  1500/15000 - Average MSE Loss: 0.108151
  Epoch  3000/15000 - Average MSE Loss: 0.046522
  Epoch  4500/15000 - Average MSE Loss: 0.006850
  Epoch  6000/15000 - Average MSE Loss: 0.002821
  Epoch  7500/15000 - Average MSE Loss: 0.001691
  Epoch  9000/15000 - Average MSE Loss: 0.001198
  Epoch 10500/15000 - Average MSE Loss: 0.000927
  Epoch 12000/15000 - Average MSE Loss: 0.000757
  Epoch 13500/15000 - Average MSE Loss: 0.000640
  Epoch 15000/15000 - Average MSE Loss: 0.000554

Training complete!

=== Final Model Predictions ===
  Input: [0.0, 0.0] -> Target: 0.0 -> Prediction: 0.026410 (Class: 0) [SUCCESS]
  Input: [0.0, 1.0] -> Target: 1.0 -> Prediction: 0.969188 (Class: 1) [SUCCESS]
  Input: [1.0, 0.0] -> Target: 1.0 -> Prediction: 0.970119 (Class: 1) [SUCCESS]
  Input: [1.0, 1.0] -> Target: 0.0 -> Prediction: 0.034502 (Class: 0) [SUCCESS]

Deallocating memory and freeing workspaces...
Cleanup complete. Program exiting successfully.
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE details.
