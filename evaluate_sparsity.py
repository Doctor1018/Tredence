"""
evaluate_sparsity.py
====================
Trains FeedForwardCIFAR10 for three λ values (low / medium / high),
then reports:
  • Sparsity level  – % of gates below threshold=0.1
  • Final test accuracy
  • Sparsity-vs-accuracy trade-off table + plots

Run:
    python evaluate_sparsity.py
"""

from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─── Hyper-parameters ──────────────────────────────────────────────────────────
LAMBDA_VALUES   = [1e-5, 1e-4, 1e-3]          # low / medium / high
LAMBDA_LABELS   = ["Low (1e-5)", "Medium (1e-4)", "High (1e-3)"]
EPOCHS          = 10
BATCH_SIZE      = 128
LR              = 1e-3
HIDDEN_SIZE     = 512
SEED            = 42
GATE_THRESHOLD  = 0.1                          # gates below this → "pruned"
DATA_DIR        = Path(".")
OUT_DIR         = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)


# ─── Model ─────────────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_scores)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)


class FeedForwardCIFAR10(nn.Module):
    def __init__(self, hidden_size=512, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            PrunableLinear(3 * 32 * 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            PrunableLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            PrunableLinear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ─── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(data_dir):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    kw = dict(root=data_dir, transform=tfm, download=True)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(datasets.CIFAR10(**kw, train=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(datasets.CIFAR10(**kw, train=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    return train_loader, test_loader


def sparsity_loss(model):
    loss = None
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            l = torch.sigmoid(m.gate_scores).sum()
            loss = l if loss is None else loss + l
    return loss if loss is not None else torch.zeros(())


@torch.no_grad()
def compute_sparsity(model, threshold=GATE_THRESHOLD):
    pruned, total = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate_scores)
            pruned += (g < threshold).sum().item()
            total  += g.numel()
    return 100.0 * pruned / total if total else 0.0


@torch.no_grad()
def collect_gates(model):
    vals = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            vals.append(torch.sigmoid(m.gate_scores).flatten().cpu())
    return torch.cat(vals) if vals else torch.empty(0)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out  = model(imgs)
        loss_sum += criterion(out, lbls).item() * lbls.size(0)
        correct  += (out.argmax(1) == lbls).sum().item()
        total    += lbls.size(0)
    return loss_sum / total, correct / total


# ─── Training loop for one λ ───────────────────────────────────────────────────
def train_lambda(lam, train_loader, test_loader, device):
    set_seed(SEED)
    model     = FeedForwardCIFAR10(HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"epoch": [], "train_acc": [], "test_acc": [], "sparsity": []}

    print(f"\n{'='*60}")
    print(f"  Training  lambda = {lam:g}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5}  {'TrainAcc':>9}  {'TestAcc':>8}  {'Sparsity':>9}")
    print("-" * 42)

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out      = model(imgs)
            ce_loss  = criterion(out, lbls)
            sp_loss  = sparsity_loss(model)
            (ce_loss + lam * sp_loss).backward()
            optimizer.step()

        _, train_acc = evaluate(model, train_loader, criterion, device)
        _, test_acc  = evaluate(model, test_loader,  criterion, device)
        sp           = compute_sparsity(model)

        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc * 100)
        history["test_acc"].append(test_acc  * 100)
        history["sparsity"].append(sp)

        print(f"{epoch:>5}  {train_acc:>9.2%}  {test_acc:>8.2%}  {sp:>8.2f}%")

    final_sp        = compute_sparsity(model)
    _, final_test   = evaluate(model, test_loader, criterion, device)
    gate_vals       = collect_gates(model)

    return {
        "lambda":      lam,
        "test_acc":    final_test,
        "sparsity":    final_sp,
        "gate_vals":   gate_vals,
        "history":     history,
    }


# ─── Plotting ──────────────────────────────────────────────────────────────────
COLORS = ["#4CAF50", "#2196F3", "#F44336"]   # green / blue / red

def plot_gate_histograms(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(f"Gate-Value Distributions (threshold = {GATE_THRESHOLD})",
                 fontsize=14, fontweight="bold")

    for ax, r, label, color in zip(axes, results, LAMBDA_LABELS, COLORS):
        vals = r["gate_vals"].numpy()
        ax.hist(vals, bins=60, range=(0, 1), color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(GATE_THRESHOLD, color="black", linestyle="--",
                   linewidth=1.5, label=f"threshold={GATE_THRESHOLD}")
        ax.set_title(f"λ = {label}\nSparsity: {r['sparsity']:.2f}%  |  "
                     f"Acc: {r['test_acc']:.2%}")
        ax.set_xlabel("Gate value σ(s)")
        ax.set_ylabel("Count" if ax is axes[0] else "")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "gate_histograms.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def plot_training_curves(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves per λ", fontsize=14, fontweight="bold")

    for r, label, color in zip(results, LAMBDA_LABELS, COLORS):
        h = r["history"]
        ax1.plot(h["epoch"], h["test_acc"],  color=color, label=label, linewidth=2)
        ax2.plot(h["epoch"], h["sparsity"],  color=color, label=label, linewidth=2)

    ax1.set_title("Test Accuracy vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Sparsity Level vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"Sparsity (% gates < {GATE_THRESHOLD})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_tradeoff(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Sparsity vs Test Accuracy Trade-off", fontsize=13, fontweight="bold")

    x = [r["sparsity"]  for r in results]
    y = [r["test_acc"] * 100 for r in results]

    for xi, yi, label, color in zip(x, y, LAMBDA_LABELS, COLORS):
        ax.scatter(xi, yi, s=120, color=color, zorder=5, label=label)
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.plot(x, y, "k--", alpha=0.4, linewidth=1.5)
    ax.set_xlabel(f"Sparsity (% gates < {GATE_THRESHOLD})")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "tradeoff.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─── Summary report ────────────────────────────────────────────────────────────
def print_report(results):
    sep = "=" * 68
    print(f"\n{sep}")
    print("  SPARSITY & ACCURACY REPORT")
    print(sep)
    print(f"  Gate threshold : {GATE_THRESHOLD}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Hidden size    : {HIDDEN_SIZE}")
    print(sep)
    print(f"{'λ':>12}  {'Label':>14}  {'Sparsity':>10}  {'Test Acc':>10}")
    print("-" * 55)
    for r, label in zip(results, LAMBDA_LABELS):
        print(f"  {r['lambda']:>10g}  {label:>14}  "
              f"{r['sparsity']:>9.2f}%  {r['test_acc']:>9.2%}")
    print(sep)

    # ── interpretation ──
    best_acc = max(results, key=lambda r: r["test_acc"])
    most_sparse = max(results, key=lambda r: r["sparsity"])
    print("\n  KEY OBSERVATIONS")
    print(f"  • Best accuracy  : λ={best_acc['lambda']:g}  → "
          f"{best_acc['test_acc']:.2%}  (sparsity {best_acc['sparsity']:.2f}%)")
    print(f"  • Most pruned    : λ={most_sparse['lambda']:g}  → "
          f"sparsity {most_sparse['sparsity']:.2f}%  "
          f"(accuracy {most_sparse['test_acc']:.2%})")
    acc_drop = (best_acc["test_acc"] - most_sparse["test_acc"]) * 100
    sp_gain  = most_sparse["sparsity"] - best_acc["sparsity"]
    print(f"  • Trade-off      : +{sp_gain:.1f}% sparsity costs "
          f"≈ {acc_drop:.2f} pp accuracy")
    print(sep)


# ─── Entry point ───────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_loader, test_loader = make_loaders(DATA_DIR)

    results = []
    for lam in LAMBDA_VALUES:
        r = train_lambda(lam, train_loader, test_loader, device)
        results.append(r)

    print_report(results)
    plot_gate_histograms(results)
    plot_training_curves(results)
    plot_tradeoff(results)

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
