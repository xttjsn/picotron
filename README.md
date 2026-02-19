# PicoTron 🤖

**Minimal Megatron-LM in < 1000 lines of Python.**

An educational implementation of the core ideas from [Megatron-LM](https://arxiv.org/abs/1909.08053) — 3D parallelism, 1F1B pipeline scheduling, ZeRO-1 optimizer, and FP32 gradient accumulation — all in a single readable Python file.

## Features

| Feature | Description |
|---------|-------------|
| **Tensor Parallelism** | Column/Row parallel linear layers that split across TP ranks |
| **Pipeline Parallelism** | Model layers sharded across pipeline stages |
| **Data Parallelism** | Gradient all-reduce across DP ranks |
| **1F1B Schedule** | One-forward-one-backward interleaved pipeline schedule |
| **ZeRO-1 Optimizer** | Optimizer states partitioned across data-parallel ranks |
| **FP32 Gradients** | FP32 gradient accumulation even with mixed precision |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    3D Parallelism Mesh                   │
│                                                         │
│  Data Parallel (DP)  ──  replicate model, avg gradients │
│  Pipeline Parallel (PP) ── shard layers across stages   │
│  Tensor Parallel (TP)  ── shard layers within a stage   │
│                                                         │
│  Rank layout: rank = dp * (PP*TP) + pp * TP + tp        │
└─────────────────────────────────────────────────────────┘
```

### 1F1B Pipeline Schedule

```
Stage 0: F F F F B F B F B   B B B
Stage 1: . F F F F B F B F B   B B
Stage 2: . . F F F F B F B F B   B
Stage 3: . . . F F F F B F B F B
         ├warmup┤├── 1F1B ──┤├cool┤
```

### ZeRO Stage 1

Each DP rank only holds optimizer states (m, v, fp32 params) for `1/DP_SIZE` of parameters. After the optimizer step, updated parameters are all-gathered.

## Files

- `picotron.py` — Core implementation (~500 lines)
- `train.py` — Single-GPU training script
- `tests/` — pytest test suite

## Quick Start

```bash
# Install
pip install torch pytest

# Train on synthetic data (CPU)
python train.py --device cpu --steps 100

# Train on GPU
python train.py --steps 200

# Run tests
pytest tests/ -v
```

## K3s Deployment

```bash
# Build and deploy
docker build -t picotron:latest .
kubectl apply -f k8s/training-job.yaml

# Watch logs
kubectl logs -f job/picotron-training
```

## Line Count

```bash
$ wc -l picotron.py
# Target: < 1000 lines
```

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
