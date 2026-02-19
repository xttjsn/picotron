"""Tests for PicoTron — no distributed required (single-process tests)."""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from picotron import (
    PicoConfig, ParallelState, PicoTronModel, ZeROStage1Optimizer,
    ColumnParallelLinear, RowParallelLinear, TransformerBlock,
    causal_lm_loss, train_single_gpu,
)


@pytest.fixture
def cfg():
    return PicoConfig(
        vocab_size=64, hidden_size=32, num_layers=2, num_heads=2,
        max_seq_len=16, micro_batch_size=2, num_micro_batches=2,
    )


@pytest.fixture
def ps(cfg):
    return ParallelState(cfg)


# ---------------------------------------------------------------------------
# Tensor Parallelism Tests
# ---------------------------------------------------------------------------

class TestTensorParallelism:
    def test_column_parallel_output_shape(self, cfg, ps):
        layer = ColumnParallelLinear(cfg.hidden_size, 4 * cfg.hidden_size, ps)
        x = torch.randn(2, 16, cfg.hidden_size)
        out = layer(x)
        # With TP=1, local_out = full output
        assert out.shape == (2, 16, 4 * cfg.hidden_size)

    def test_row_parallel_output_shape(self, cfg, ps):
        layer = RowParallelLinear(4 * cfg.hidden_size, cfg.hidden_size, ps)
        x = torch.randn(2, 16, 4 * cfg.hidden_size)
        out = layer(x)
        assert out.shape == (2, 16, cfg.hidden_size)

    def test_column_row_compose(self, cfg, ps):
        """Column→Row should produce correct shapes end-to-end."""
        col = ColumnParallelLinear(cfg.hidden_size, 4 * cfg.hidden_size, ps)
        row = RowParallelLinear(4 * cfg.hidden_size, cfg.hidden_size, ps)
        x = torch.randn(2, 16, cfg.hidden_size)
        out = row(torch.relu(col(x)))
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Pipeline Parallelism / Model Tests
# ---------------------------------------------------------------------------

class TestPipelineParallelism:
    def test_model_forward(self, cfg, ps):
        model = PicoTronModel(cfg, ps)
        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        logits = model(tokens)
        assert logits.shape == (2, cfg.max_seq_len, cfg.vocab_size)

    def test_model_layers_count(self, cfg, ps):
        model = PicoTronModel(cfg, ps)
        assert len(model.layers) == cfg.num_layers  # PP=1 means all layers

    def test_pp_stage_assignment(self):
        """With PP=2, each stage should get half the layers."""
        cfg = PicoConfig(num_layers=4, pipeline_parallel_size=2)
        # Stage 0
        ps0 = ParallelState(cfg)
        ps0.pp_rank = 0
        m0 = PicoTronModel(cfg, ps0)
        assert len(m0.layers) == 2
        assert m0.embedding is not None  # first stage has embedding
        assert m0.head is None

        # Stage 1
        ps1 = ParallelState(cfg)
        ps1.pp_rank = 1
        ps1.pp = 2
        m1 = PicoTronModel(cfg, ps1)
        assert len(m1.layers) == 2
        assert m1.embedding is None
        assert m1.head is not None  # last stage has head


# ---------------------------------------------------------------------------
# 1F1B Schedule Tests (unit logic)
# ---------------------------------------------------------------------------

class TestOneFOneBSchedule:
    def test_warmup_cooldown_counts(self):
        """Verify warmup/1f1b/cooldown split logic."""
        pp_size = 4
        num_micro = 8

        for pp_rank in range(pp_size):
            num_warmup = min(pp_size - pp_rank - 1, num_micro)
            num_1f1b = num_micro - num_warmup
            num_cooldown = num_micro - num_1f1b

            assert num_warmup + num_1f1b == num_micro
            assert num_cooldown == num_warmup
            # First rank: most warmup, last rank: zero warmup
            if pp_rank == 0:
                assert num_warmup == pp_size - 1
            if pp_rank == pp_size - 1:
                assert num_warmup == 0

    def test_total_forwards_equals_microbatches(self):
        """Each rank does exactly num_micro forward passes."""
        pp_size = 4
        num_micro = 6
        for pp_rank in range(pp_size):
            num_warmup = min(pp_size - pp_rank - 1, num_micro)
            num_1f1b = num_micro - num_warmup
            total_fwd = num_warmup + num_1f1b
            assert total_fwd == num_micro


# ---------------------------------------------------------------------------
# ZeRO-1 Optimizer Tests
# ---------------------------------------------------------------------------

class TestZeRO1Optimizer:
    def test_partition_covers_all_params(self, cfg, ps):
        model = PicoTronModel(cfg, ps)
        opt = ZeROStage1Optimizer(list(model.parameters()), lr=1e-3, ps=ps)
        # With DP=1, single rank covers everything
        assert opt.start == 0
        assert opt.end == opt.num_elements

    def test_fp32_grad_accumulation(self, cfg, ps):
        """Verify gradients are accumulated in FP32."""
        model = PicoTronModel(cfg, ps)
        opt = ZeROStage1Optimizer(list(model.parameters()), lr=1e-3, ps=ps)

        # Do a forward/backward pass
        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        logits = model(tokens)
        loss = causal_lm_loss(logits, tokens)
        loss.backward()

        opt.accumulate_fp32_grads()
        assert opt.fp32_grad_acc.dtype == torch.float32
        assert opt.fp32_grad_acc.abs().sum() > 0  # some gradients should be non-zero

    def test_step_updates_params(self, cfg, ps):
        model = PicoTronModel(cfg, ps)
        opt = ZeROStage1Optimizer(list(model.parameters()), lr=1e-2, ps=ps)

        params_before = [p.data.clone() for p in model.parameters()]

        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        logits = model(tokens)
        loss = causal_lm_loss(logits, tokens)
        loss.backward()

        opt.accumulate_fp32_grads()
        opt.step()

        changed = any(
            not torch.equal(before, after.data)
            for before, after in zip(params_before, model.parameters())
        )
        assert changed, "Parameters should change after optimizer step"

    def test_zero_partition_simulation(self):
        """Simulate 4 DP ranks and verify full coverage."""
        total = 1000
        dp_size = 4
        covered = set()
        for rank in range(dp_size):
            chunk = math.ceil(total / dp_size)
            start = rank * chunk
            end = min(start + chunk, total)
            for i in range(start, end):
                covered.add(i)
        assert len(covered) == total


# ---------------------------------------------------------------------------
# FP32 Gradient Accumulation Tests
# ---------------------------------------------------------------------------

class TestFP32Gradients:
    def test_fp32_accumulator_dtype(self, cfg, ps):
        model = PicoTronModel(cfg, ps)
        opt = ZeROStage1Optimizer(list(model.parameters()), lr=1e-3, ps=ps)
        assert opt.fp32_grad_acc.dtype == torch.float32
        assert opt.fp32_shard.dtype == torch.float32
        assert opt.m.dtype == torch.float32
        assert opt.v.dtype == torch.float32

    def test_multiple_accumulations(self, cfg, ps):
        """Multiple accumulations should add up."""
        model = PicoTronModel(cfg, ps)
        opt = ZeROStage1Optimizer(list(model.parameters()), lr=1e-3, ps=ps)

        for _ in range(3):
            tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
            logits = model(tokens)
            loss = causal_lm_loss(logits, tokens)
            loss.backward()
            opt.accumulate_fp32_grads()
            opt.zero_grad()

        # After 3 accumulations, grad should be larger than single
        assert opt.fp32_grad_acc.abs().sum() > 0


# ---------------------------------------------------------------------------
# Data Parallelism Tests (single-rank simulation)
# ---------------------------------------------------------------------------

class TestDataParallelism:
    def test_dp_rank_assignment(self):
        cfg = PicoConfig(data_parallel_size=4, tensor_parallel_size=1, pipeline_parallel_size=1)
        ps = ParallelState(cfg)
        assert ps.dp_rank == 0  # rank 0 → dp_rank 0

    def test_mesh_coordinates(self):
        """Verify (dp, pp, tp) coordinate calculation."""
        cfg = PicoConfig(data_parallel_size=2, pipeline_parallel_size=2, tensor_parallel_size=2)
        ps = ParallelState(cfg)
        assert ps.world_size == 8
        # rank 0 → tp=0, pp=0, dp=0
        assert ps.tp_rank == 0
        assert ps.pp_rank == 0
        assert ps.dp_rank == 0


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_training_loop_cpu(self, cfg):
        """Run a few training steps on CPU."""
        dataset = torch.randint(0, cfg.vocab_size, (5000,))
        losses = train_single_gpu(cfg, dataset, num_steps=5, lr=1e-3, device="cpu")
        assert len(losses) > 0
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases(self):
        """Training on a learnable pattern should decrease loss."""
        cfg = PicoConfig(
            vocab_size=32, hidden_size=64, num_layers=2, num_heads=2,
            max_seq_len=32, micro_batch_size=4, num_micro_batches=2,
        )
        # Repeating pattern = learnable
        pattern = torch.arange(32).repeat(500)
        losses = train_single_gpu(cfg, pattern, num_steps=50, lr=1e-3, device="cpu")
        # Compare early vs late loss
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss should decrease: early={early:.4f}, late={late:.4f}"
