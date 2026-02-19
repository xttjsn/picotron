"""
PicoTron — Minimal Megatron-LM in < 1000 lines of Python.

Implements:
  - 3D Parallelism (Tensor ∥, Pipeline ∥, Data ∥)
  - 1F1B Pipeline Schedule
  - ZeRO-1 Optimizer (partitioned optimizer states)
  - FP32 Gradient Accumulation with mixed precision
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PicoConfig:
    vocab_size: int = 512
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    # Training
    micro_batch_size: int = 4
    num_micro_batches: int = 4  # gradient accumulation steps
    fp16: bool = False


# ---------------------------------------------------------------------------
# 3D Parallelism — Process Group Helpers
# ---------------------------------------------------------------------------

class ParallelState:
    """Manages the 3D mesh of (DP, PP, TP) process groups."""

    def __init__(self, cfg: PicoConfig):
        self.tp = cfg.tensor_parallel_size
        self.pp = cfg.pipeline_parallel_size
        self.dp = cfg.data_parallel_size
        self.world_size = self.tp * self.pp * self.dp

        self.tp_group: Optional[dist.ProcessGroup] = None
        self.pp_group: Optional[dist.ProcessGroup] = None
        self.dp_group: Optional[dist.ProcessGroup] = None

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        # Coordinates in (dp, pp, tp) mesh
        self.tp_rank = self.rank % self.tp
        self.pp_rank = (self.rank // self.tp) % self.pp
        self.dp_rank = self.rank // (self.tp * self.pp)

    def init_groups(self):
        """Create process groups for each parallelism dimension."""
        if not dist.is_initialized():
            return
        # TP groups: ranks that share the same (dp, pp) but differ in tp
        for dp in range(self.dp):
            for pp in range(self.pp):
                ranks = [dp * self.pp * self.tp + pp * self.tp + tp for tp in range(self.tp)]
                g = dist.new_group(ranks)
                if self.dp_rank == dp and self.pp_rank == pp:
                    self.tp_group = g

        # PP groups: ranks that share the same (dp, tp) but differ in pp
        for dp in range(self.dp):
            for tp in range(self.tp):
                ranks = [dp * self.pp * self.tp + pp * self.tp + tp for pp in range(self.pp)]
                g = dist.new_group(ranks)
                if self.dp_rank == dp and self.tp_rank == tp:
                    self.pp_group = g

        # DP groups: ranks that share the same (pp, tp) but differ in dp
        for pp in range(self.pp):
            for tp in range(self.tp):
                ranks = [dp * self.pp * self.tp + pp * self.tp + tp for dp in range(self.dp)]
                g = dist.new_group(ranks)
                if self.pp_rank == pp and self.tp_rank == tp:
                    self.dp_group = g

    @property
    def is_first_pipeline_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_pipeline_stage(self) -> bool:
        return self.pp_rank == self.pp - 1

    @property
    def pp_prev_rank(self) -> int:
        return self.rank - self.tp

    @property
    def pp_next_rank(self) -> int:
        return self.rank + self.tp


# ---------------------------------------------------------------------------
# Tensor Parallelism Primitives
# ---------------------------------------------------------------------------

class _CopyToTP(torch.autograd.Function):
    """Forward: identity. Backward: all-reduce gradients across TP group."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    """Forward: all-reduce across TP group. Backward: identity."""
    @staticmethod
    def forward(ctx, x, group):
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def copy_to_tp(x: torch.Tensor, group) -> torch.Tensor:
    if group is None:
        return x
    return _CopyToTP.apply(x, group)


def reduce_from_tp(x: torch.Tensor, group) -> torch.Tensor:
    if group is None:
        return x
    return _ReduceFromTP.apply(x, group)


class ColumnParallelLinear(nn.Module):
    """Linear layer split along output dim across TP ranks."""

    def __init__(self, in_f: int, out_f: int, ps: ParallelState, bias: bool = True):
        super().__init__()
        self.ps = ps
        assert out_f % ps.tp == 0
        self.local_out = out_f // ps.tp
        self.weight = nn.Parameter(torch.empty(self.local_out, in_f))
        self.bias = nn.Parameter(torch.empty(self.local_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tp(x, self.ps.tp_group)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear layer split along input dim across TP ranks."""

    def __init__(self, in_f: int, out_f: int, ps: ParallelState, bias: bool = True):
        super().__init__()
        self.ps = ps
        assert in_f % ps.tp == 0
        self.local_in = in_f // ps.tp
        self.weight = nn.Parameter(torch.empty(out_f, self.local_in))
        self.bias = nn.Parameter(torch.empty(out_f)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        out = reduce_from_tp(out, self.ps.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: PicoConfig, ps: ParallelState):
        super().__init__()
        self.cfg = cfg
        self.ps = ps
        assert cfg.num_heads % ps.tp == 0
        self.local_heads = cfg.num_heads // ps.tp
        self.head_dim = cfg.hidden_size // cfg.num_heads

        self.qkv = ColumnParallelLinear(cfg.hidden_size, 3 * cfg.hidden_size, ps, bias=False)
        self.out_proj = RowParallelLinear(cfg.hidden_size, cfg.hidden_size, ps, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x)  # (B, S, 3 * local_hidden)
        qkv = qkv.reshape(B, S, 3, self.local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, local_heads, head_dim)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, heads, S, dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, S, dim)
        out = out.transpose(1, 2).reshape(B, S, self.local_heads * self.head_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PicoConfig, ps: ParallelState):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = MultiHeadAttention(cfg, ps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp_up = ColumnParallelLinear(cfg.hidden_size, 4 * cfg.hidden_size, ps)
        self.mlp_down = RowParallelLinear(4 * cfg.hidden_size, cfg.hidden_size, ps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = F.gelu(self.mlp_up(self.ln2(x)))
        x = x + self.dropout(self.mlp_down(h))
        return x


# ---------------------------------------------------------------------------
# Pipeline-Parallel Model: each rank holds a shard of layers
# ---------------------------------------------------------------------------

class PicoTronModel(nn.Module):
    """GPT-like model supporting tensor + pipeline parallelism."""

    def __init__(self, cfg: PicoConfig, ps: ParallelState):
        super().__init__()
        self.cfg = cfg
        self.ps = ps

        # Divide layers across pipeline stages
        assert cfg.num_layers % ps.pp == 0
        layers_per_stage = cfg.num_layers // ps.pp
        start = ps.pp_rank * layers_per_stage
        end = start + layers_per_stage

        # Embedding & head only on first/last stage
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size) if ps.is_first_pipeline_stage else None
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_size) if ps.is_first_pipeline_stage else None

        self.layers = nn.ModuleList([TransformerBlock(cfg, ps) for _ in range(start, end)])

        self.ln_f = nn.LayerNorm(cfg.hidden_size) if ps.is_last_pipeline_stage else None
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) if ps.is_last_pipeline_stage else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: token ids (B, S) on first stage, hidden (B, S, H) on others."""
        if self.embedding is not None:
            B, S = x.shape
            positions = torch.arange(S, device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        if self.ln_f is not None:
            x = self.ln_f(x)
            x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# 1F1B Pipeline Schedule
# ---------------------------------------------------------------------------

class PipelineSchedule:
    """One-forward-one-backward (1F1B) schedule for pipeline parallelism."""

    def __init__(self, model: PicoTronModel, ps: ParallelState, cfg: PicoConfig):
        self.model = model
        self.ps = ps
        self.cfg = cfg
        self.num_micro = cfg.num_micro_batches
        self.pp_size = ps.pp

    def _send(self, tensor: torch.Tensor, dst: int):
        dist.send(tensor, dst, group=self.ps.pp_group)

    def _recv(self, shape: tuple, src: int, dtype=torch.float32) -> torch.Tensor:
        buf = torch.empty(shape, device="cuda", dtype=dtype)
        dist.recv(buf, src, group=self.ps.pp_group)
        return buf

    def _forward_step(self, micro_batch, input_tensor=None):
        if self.ps.is_first_pipeline_stage:
            output = self.model(micro_batch)
        else:
            output = self.model(input_tensor)
        return output

    def _backward_step(self, output_tensor, output_grad=None):
        if output_grad is None:
            output_tensor.backward()
        else:
            output_tensor.backward(output_grad)

    def run(self, micro_batches: List[torch.Tensor], loss_fn) -> float:
        """Execute 1F1B schedule. Returns average loss (on last stage)."""
        num_warmup = self.pp_size - self.ps.pp_rank - 1
        num_warmup = min(num_warmup, self.num_micro)
        num_1f1b = self.num_micro - num_warmup
        num_cooldown = self.num_micro - num_1f1b

        input_tensors: List[Optional[torch.Tensor]] = []
        output_tensors: List[Optional[torch.Tensor]] = []
        total_loss = 0.0
        H = self.cfg.hidden_size
        B = self.cfg.micro_batch_size
        S = self.cfg.max_seq_len

        def recv_forward():
            if self.ps.is_first_pipeline_stage:
                return None
            return self._recv((B, S, H), self.ps.pp_prev_rank)

        def send_forward(out):
            if self.ps.is_last_pipeline_stage:
                return
            self._send(out.detach(), self.ps.pp_next_rank)

        def recv_backward():
            if self.ps.is_last_pipeline_stage:
                return None
            return self._recv((B, S, H), self.ps.pp_next_rank)

        def send_backward(inp):
            if self.ps.is_first_pipeline_stage:
                return
            self._send(inp.grad.detach(), self.ps.pp_prev_rank)

        # --- Warmup forward passes ---
        for i in range(num_warmup):
            inp = recv_forward()
            if inp is not None:
                inp.requires_grad_(True)
            out = self._forward_step(micro_batches[i], inp)
            if self.ps.is_last_pipeline_stage:
                loss = loss_fn(out, micro_batches[i])
                total_loss += loss.item()
                output_tensors.append(loss)
            else:
                send_forward(out)
                output_tensors.append(out)
            input_tensors.append(inp)

        # --- Steady 1F1B ---
        for i in range(num_1f1b):
            mb_idx = num_warmup + i
            inp = recv_forward()
            if inp is not None:
                inp.requires_grad_(True)
            out = self._forward_step(micro_batches[mb_idx], inp)
            if self.ps.is_last_pipeline_stage:
                loss = loss_fn(out, micro_batches[mb_idx])
                total_loss += loss.item()
                output_tensors.append(loss)
            else:
                send_forward(out)
                output_tensors.append(out)
            input_tensors.append(inp)

            # Backward for oldest warmup microbatch
            out_grad = recv_backward()
            bwd_out = output_tensors.pop(0)
            bwd_inp = input_tensors.pop(0)
            self._backward_step(bwd_out, out_grad)
            if bwd_inp is not None:
                send_backward(bwd_inp)

        # --- Cooldown backward passes ---
        for i in range(num_cooldown):
            out_grad = recv_backward()
            bwd_out = output_tensors.pop(0)
            bwd_inp = input_tensors.pop(0)
            self._backward_step(bwd_out, out_grad)
            if bwd_inp is not None:
                send_backward(bwd_inp)

        return total_loss / self.num_micro if self.ps.is_last_pipeline_stage else 0.0


# ---------------------------------------------------------------------------
# ZeRO Stage 1 Optimizer
# ---------------------------------------------------------------------------

class ZeROStage1Optimizer:
    """
    Partitions optimizer states across data-parallel ranks.
    Each rank owns a shard of parameters and only maintains optimizer state for that shard.
    FP32 gradient accumulation: keeps FP32 copies of gradients.
    """

    def __init__(self, params: List[nn.Parameter], lr: float, ps: ParallelState,
                 weight_decay: float = 0.01, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.ps = ps
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0

        # Flatten all parameters into one buffer for partitioning
        self.all_params = list(params)
        self.flat_params = self._flatten(self.all_params)
        self.num_elements = self.flat_params.numel()

        # Partition across DP ranks
        dp_size = max(ps.dp, 1)
        dp_rank = ps.dp_rank
        chunk = math.ceil(self.num_elements / dp_size)
        self.start = dp_rank * chunk
        self.end = min(self.start + chunk, self.num_elements)
        self.local_size = self.end - self.start

        # FP32 copies for this shard
        self.fp32_shard = self.flat_params[self.start:self.end].float().clone()
        self.fp32_grad_acc = torch.zeros_like(self.fp32_shard)  # FP32 gradient accumulator
        self.m = torch.zeros_like(self.fp32_shard)
        self.v = torch.zeros_like(self.fp32_shard)

        # Map from flat index ranges to original parameters
        self._build_param_map()

    def _flatten(self, params: List[nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in params])

    def _build_param_map(self):
        """Build mapping: which params overlap with our shard."""
        self.param_offsets = []
        offset = 0
        for p in self.all_params:
            n = p.numel()
            self.param_offsets.append((offset, offset + n, p))
            offset += n

    def zero_grad(self):
        for p in self.all_params:
            if p.grad is not None:
                p.grad.zero_()

    def accumulate_fp32_grads(self):
        """Gather gradients into FP32 accumulator for our shard."""
        flat_grad = torch.cat([
            p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device)
            for p in self.all_params
        ])
        # All-reduce grads across DP group
        if self.ps.dp_group is not None and self.ps.dp > 1:
            dist.all_reduce(flat_grad, group=self.ps.dp_group)
            flat_grad.div_(self.ps.dp)

        self.fp32_grad_acc.add_(flat_grad[self.start:self.end].float())

    def step(self):
        """Adam step on our shard, then all-gather updated params."""
        self.step_count += 1
        grad = self.fp32_grad_acc

        # Adam
        self.m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_hat = self.v / (1 - self.beta2 ** self.step_count)

        self.fp32_shard.add_(
            -(self.lr * m_hat / (v_hat.sqrt() + self.eps))
            - self.lr * self.weight_decay * self.fp32_shard
        )

        # Reset accumulator
        self.fp32_grad_acc.zero_()

        # Write shard back and all-gather
        self._write_back()

    def _write_back(self):
        """Broadcast updated shard to all DP ranks and write back to params."""
        full = torch.zeros(self.num_elements, device=self.fp32_shard.device, dtype=self.fp32_shard.dtype)
        full[self.start:self.end] = self.fp32_shard

        if self.ps.dp_group is not None and self.ps.dp > 1:
            dist.all_reduce(full, group=self.ps.dp_group)

        # Write back to original params
        offset = 0
        for p in self.all_params:
            n = p.numel()
            p.data.copy_(full[offset:offset + n].reshape(p.shape).to(p.dtype))
            offset += n


# ---------------------------------------------------------------------------
# Loss function helper for pipeline
# ---------------------------------------------------------------------------

def causal_lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for causal LM: predict next token."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


# ---------------------------------------------------------------------------
# Single-GPU / non-distributed training loop (for testing & demo)
# ---------------------------------------------------------------------------

def train_single_gpu(cfg: PicoConfig, dataset: torch.Tensor, num_steps: int = 100,
                     lr: float = 3e-4, device: str = "cuda") -> List[float]:
    """Simple training loop without distributed (for single GPU / CPU)."""
    ps = ParallelState(cfg)
    model = PicoTronModel(cfg, ps).to(device)
    optimizer = ZeROStage1Optimizer(list(model.parameters()), lr=lr, ps=ps)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()

        # Sample micro-batches
        for _ in range(cfg.num_micro_batches):
            idx = torch.randint(0, len(dataset) - cfg.max_seq_len, (cfg.micro_batch_size,))
            batch = torch.stack([dataset[i:i + cfg.max_seq_len] for i in idx]).to(device)
            logits = model(batch)
            loss = causal_lm_loss(logits, batch) / cfg.num_micro_batches
            loss.backward()
            losses.append(loss.item() * cfg.num_micro_batches)

        optimizer.accumulate_fp32_grads()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {losses[-1]:.4f}")

    return losses
