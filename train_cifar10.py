import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class Config:
    data_dir: Path = Path("data")
    plot_dir: Path = Path("plots")
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    lambda_values: Tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    hidden_size: int = 512
    num_workers: int = 2
    seed: int = 42
    download: bool = True


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_scores)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)


class FeedForwardCIFAR10(nn.Module):
    def __init__(self, hidden_size: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            PrunableLinear(3 * 32 * 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            PrunableLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            PrunableLinear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    loss = None
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            layer_loss = torch.sigmoid(module.gate_scores).sum()
            loss = layer_loss if loss is None else loss + layer_loss

    if loss is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return loss


@torch.no_grad()
def compute_sparsity(model: nn.Module, threshold: float = 0.1) -> float:
    pruned_gates = 0
    total_gates = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            pruned_gates += (gates < threshold).sum().item()
            total_gates += gates.numel()

    if total_gates == 0:
        return 0.0
    return 100.0 * pruned_gates / total_gates


@torch.no_grad()
def collect_gate_values(model: nn.Module) -> torch.Tensor:
    gate_values = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().flatten().cpu()
            gate_values.append(gates)

    if not gate_values:
        return torch.empty(0)
    return torch.cat(gate_values)


def plot_gate_histogram(
    gate_values: torch.Tensor,
    sparsity_lambda: float,
    output_dir: Path,
    bins: int = 50,
) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate_histogram_lambda_{sparsity_lambda:g}.png"

    plt.figure(figsize=(8, 5))
    plt.hist(gate_values.numpy(), bins=bins, range=(0.0, 1.0), edgecolor="black")
    plt.xlabel("Gate value: sigmoid(gate_scores)")
    plt.ylabel("Count")
    plt.title(f"Gate Value Histogram (lambda={sparsity_lambda:g})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=config.download,
    )
    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        transform=transform,
        download=config.download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sparsity_lambda: float,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_sparsity_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        cross_entropy_loss = criterion(outputs, labels)
        gate_sparsity_loss = sparsity_loss(model)
        total_loss = cross_entropy_loss + sparsity_lambda * gate_sparsity_loss
        total_loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += total_loss.item() * batch_size
        running_sparsity_loss += gate_sparsity_loss.item() * batch_size
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total, running_sparsity_loss / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def train_for_lambda(
    config: Config,
    sparsity_lambda: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, float, float]:
    set_seed(config.seed)
    model = FeedForwardCIFAR10(hidden_size=config.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"\nTraining with lambda={sparsity_lambda:g}")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, train_sparsity_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            sparsity_lambda=sparsity_lambda,
        )
        test_loss, test_acc = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        sparsity_percent = compute_sparsity(model)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.2%} | "
            f"sparsity loss: {train_sparsity_loss:.2f} | "
            f"sparsity: {sparsity_percent:.2f}% | "
            f"test loss: {test_loss:.4f}, test acc: {test_acc:.2%}"
        )

    final_test_loss, final_test_acc = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    final_sparsity = compute_sparsity(model)
    gate_values = collect_gate_values(model)
    histogram_path = plot_gate_histogram(
        gate_values=gate_values,
        sparsity_lambda=sparsity_lambda,
        output_dir=config.plot_dir,
    )
    print(
        f"Final lambda={sparsity_lambda:g} | "
        f"test accuracy: {final_test_acc:.2%} | "
        f"sparsity: {final_sparsity:.2f}% | "
        f"histogram: {histogram_path}"
    )
    return model, final_test_acc, final_sparsity


def train(config: Config) -> None:
    set_seed(config.seed)
    device = get_device()
    train_loader, test_loader = create_data_loaders(config)

    print(f"Using device: {device}")
    print(f"Classes: {', '.join(CIFAR10_CLASSES)}")
    print(f"Lambda values: {', '.join(f'{value:g}' for value in config.lambda_values)}")

    results = []
    for sparsity_lambda in config.lambda_values:
        _, test_acc, sparsity_percent = train_for_lambda(
            config=config,
            sparsity_lambda=sparsity_lambda,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        results.append((sparsity_lambda, test_acc, sparsity_percent))

    print("\nSummary")
    for sparsity_lambda, test_acc, sparsity_percent in results:
        print(
            f"lambda={sparsity_lambda:g} | "
            f"test accuracy: {test_acc:.2%} | "
            f"sparsity: {sparsity_percent:.2f}%"
        )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a simple feedforward neural network on CIFAR-10."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3],
        help="Lambda values to sweep for the sparsity loss.",
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use an existing CIFAR-10 dataset under --data-dir.",
    )
    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        plot_dir=args.plot_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_values=tuple(args.lambda_values),
        hidden_size=args.hidden_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=not args.no_download,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
