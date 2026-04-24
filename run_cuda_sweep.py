from pathlib import Path

import torch

from train_cifar10 import Config, train


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch build or use a GPU machine.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    config = Config(
        data_dir=Path("."),
        plot_dir=Path("plots_cuda"),
        batch_size=128,
        epochs=10,
        learning_rate=1e-3,
        lambda_values=(1e-5, 1e-4, 1e-3),
        hidden_size=512,
        num_workers=2,
        seed=42,
        download=True,
    )
    train(config)


if __name__ == "__main__":
    main()
