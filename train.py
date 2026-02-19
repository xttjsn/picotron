#!/usr/bin/env python3
"""
PicoTron training script — works on a single GPU with a toy character-level dataset.
Usage: python train.py [--steps 200] [--device cuda]
"""

import argparse
import torch
from picotron import PicoConfig, train_single_gpu


def make_toy_dataset(vocab_size: int = 512, length: int = 50_000) -> torch.Tensor:
    """Generate a synthetic dataset: repeating patterns so the model can learn something."""
    # Create a pattern that has learnable structure
    pattern = torch.randint(0, vocab_size, (1000,))
    repeats = length // len(pattern) + 1
    data = pattern.repeat(repeats)[:length]
    return data


def load_text_dataset(path: str, vocab_size: int = 512) -> torch.Tensor:
    """Load a text file as byte-level tokens (mod vocab_size)."""
    with open(path, "r") as f:
        text = f.read()
    tokens = torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)
    return tokens


def main():
    parser = argparse.ArgumentParser(description="PicoTron Training")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--num-micro-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    args = parser.parse_args()

    cfg = PicoConfig(
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_heads=args.heads,
        max_seq_len=args.seq_len,
        micro_batch_size=args.micro_batch,
        num_micro_batches=args.num_micro_batches,
    )

    print(f"PicoTron Training — device={args.device}")
    print(f"  Config: hidden={cfg.hidden_size}, layers={cfg.num_layers}, "
          f"heads={cfg.num_heads}, seq_len={cfg.max_seq_len}")

    if args.data:
        dataset = load_text_dataset(args.data, cfg.vocab_size)
    else:
        dataset = make_toy_dataset(cfg.vocab_size)
    print(f"  Dataset: {len(dataset)} tokens")

    losses = train_single_gpu(cfg, dataset, num_steps=args.steps, lr=args.lr, device=args.device)
    print(f"\nTraining complete. Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
