# Credance — Self-Pruning Neural Networks on CIFAR-10

A PyTorch implementation of **learnable gate-based weight pruning** for image classification.  
The network automatically learns which weights to prune during training via differentiable gate parameters.

---

## Method

Each linear layer is replaced with a `PrunableLinear` layer containing:

- **`weight`** — the standard weight matrix  
- **`gate_scores`** — a learnable parameter of the same shape  

During the forward pass, an element-wise gate is applied:

```
gate      = sigmoid(gate_scores)          # ∈ (0, 1)
effective = weight × gate                 # pruned weight
```

The training loss combines cross-entropy with an **L1 sparsity regulariser**:

```
L_total = L_CE  +  λ · Σ sigmoid(gate_scores)
```

A high λ pushes gates toward 0, pruning more weights.  
A low λ preserves accuracy but prunes fewer weights.

---

## Architecture

```
Input (3×32×32)
  └─ Flatten → 3072
       └─ PrunableLinear(3072 → 512) + ReLU + Dropout(0.2)
            └─ PrunableLinear(512 → 256) + ReLU + Dropout(0.2)
                 └─ PrunableLinear(256 → 10)  [logits]
```

---

## Results (λ sweep)

| λ (lambda) | Sparsity (gates < 0.1) | Final Test Accuracy |
|:----------:|:----------------------:|:-------------------:|
| 1e-5 (Low) | ~low%                  | ~high%              |
| 1e-4 (Med) | ~medium%               | ~medium%            |
| 1e-3 (High)| ~high%                 | ~lower%             |

> Run `evaluate_sparsity.py` to reproduce exact numbers on your machine.

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision matplotlib

# 2. Train + evaluate all three lambda values
python evaluate_sparsity.py

# 3. (Optional) CUDA sweep only
python run_cuda_sweep.py
```

Outputs are saved to `eval_results/`:
- `gate_histograms.png` — gate-value distributions per λ  
- `training_curves.png` — accuracy & sparsity over epochs  
- `tradeoff.png`        — sparsity vs accuracy scatter  

---

## Files

| File | Description |
|------|-------------|
| `train_cifar10.py` | Core model, training loop, helpers |
| `evaluate_sparsity.py` | Full λ sweep + report + plots |
| `run_cuda_sweep.py` | GPU-only convenience runner |

---

## Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- torchvision  
- matplotlib  

---

## License

MIT
