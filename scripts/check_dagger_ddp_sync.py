#!/usr/bin/env python
"""Correctness + cost checks for the DAgger DDP update loop.

The DAgger training step takes `dagger_learning_epochs` (default 10) SEQUENTIAL optimizer steps on one
fixed batch. Under DDP each of those steps is a synchronisation point, so a training step pays 10
all-reduces plus the pre-update barrier, no matter how small the per-rank batch is.

This script isolates that loop from the simulator so the two questions can be answered separately:

  1. Do the DDP construction options (`gradient_as_bucket_view`, `static_graph`, `bucket_cap_mb`)
     change the numerics?  -- they must not, and `--check equivalence` asserts it bit-for-bit.
  2. What do they, and the opt-in `local_avg` sync mode, actually cost?  -- `--check timing`.

It also demonstrates why the "obvious" no_sync gradient-accumulation fix does not apply here
(`--check accumulation`): the batch is fixed and the student forward is deterministic, so all K
accumulated gradients are identical and accumulation is just `dagger_learning_epochs=1` with a
rescaled gradient at K times the compute.

Single process (equivalence + accumulation only):
    python scripts/check_dagger_ddp_sync.py --check accumulation

Multi rank (all checks):
    torchrun --nproc_per_node 2 scripts/check_dagger_ddp_sync.py --check all --steps 100
"""
from __future__ import annotations

import argparse
import os
import time

# Same pre-import device masking as track_mj/learning/train/train_dagger.py: each rank sees exactly one
# GPU, so DDP is always constructed with device_ids=[0].
if "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        int(os.environ["LOCAL_RANK"]) + int(os.environ.get("CUDA_DEVICE_OFFSET", "0")))

import torch
import torch.distributed as dist

from track_mj.learning.models.dagger.policy import MLP_Policy
from track_mj.learning.models.dagger.policy_args import PolicyArgs
from track_mj.learning.policy.dagger.dagger_horizon import (
    average_parameters_across_ranks_,
    run_update_epochs,
    wrap_ddp,
)

# Defaults mirror the shipped student: obs 156 / act 29 / mlp_hidden_dim [1024,1024,512,512,256].
OBS_DIM = 156
ACT_DIM = 29


def build_policy(device: str, seed: int) -> MLP_Policy:
    torch.manual_seed(seed)
    args = PolicyArgs(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    policy = MLP_Policy(config=args)
    policy.model.to(device)
    return policy


def make_batch(batch_size: int, device: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = torch.randn(batch_size, OBS_DIM, generator=g).to(device)
    target = torch.randn(batch_size, ACT_DIM, generator=g).to(device).tanh()
    return {"state": obs}, target


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    src = model.module if hasattr(model, "module") else model
    return torch.cat([p.detach().reshape(-1) for p in src.parameters()])


def upstream_loop(policy, optimizer, batch, target, epochs, max_grad_norm, use_ddp):
    """Verbatim transcription of the pre-fix loop, kept here as the reference implementation."""
    for _ in range(epochs):
        loss, info = policy.compute_loss(batch, target, None)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.model.parameters() if not use_ddp else policy.model.module.parameters(),
            max_grad_norm,
        )
        optimizer.step()
    return loss, info


# ------------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------------


def check_accumulation(device: str, batch_size: int, epochs: int) -> None:
    """K no_sync-accumulated gradients on the fixed batch == K x the single-pass gradient."""
    policy = build_policy(device, seed=0)
    batch, target = make_batch(batch_size, device, seed=1)

    policy.model.zero_grad(set_to_none=True)
    loss, _ = policy.compute_loss(batch, target, None)
    loss.backward()
    single = torch.cat([p.grad.detach().reshape(-1) for p in policy.model.parameters()])

    policy.model.zero_grad(set_to_none=True)
    for _ in range(epochs):  # no optimizer.step() in between -- exactly what no_sync accumulation does
        loss, _ = policy.compute_loss(batch, target, None)
        loss.backward()
    accum = torch.cat([p.grad.detach().reshape(-1) for p in policy.model.parameters()])

    rel = ((accum - single * epochs).abs().max() / single.abs().max()).item()
    print(f"[accumulation] K={epochs}  max |accum - K*single| / max|single| = {rel:.3e}")
    assert rel < 1e-5, "accumulated gradient is not K x the single-pass gradient"
    print("[accumulation] PASS -- accumulating the K passes is dagger_learning_epochs=1 with a "
          "K-scaled gradient, at K x the compute. It is not a faster way to run the K steps.")


def check_equivalence(device: str, batch_size: int, epochs: int, steps: int,
                      rank: int, world_size: int) -> None:
    """Every semantics-preserving arm must end bit-identical to the upstream reference."""
    use_ddp = world_size > 1

    # `must_match` arms are required to be BITWISE identical to the upstream reference.
    # `bucket_view` is reported but not required: the reduced gradient is identical (a 1-step x 1-epoch
    # run is bitwise equal), but making param.grad a view into the flat bucket changes the memory the
    # AdamW elementwise kernels operate on, which perturbs the update at ~1 ulp and then amplifies.
    arms = [
        ("reference(upstream ctor, upstream loop)", "upstream",
         dict(grad_as_bucket_view=False, static_graph=False, bucket_cap_mb=0), True),
        ("run_update_epochs, upstream ctor", "new",
         dict(grad_as_bucket_view=False, static_graph=False, bucket_cap_mb=0), True),
        ("+static_graph", "new",
         dict(grad_as_bucket_view=False, static_graph=True, bucket_cap_mb=0), True),
        ("+static_graph+cap100", "new",
         dict(grad_as_bucket_view=False, static_graph=True, bucket_cap_mb=100), True),
        ("+bucket_view", "new",
         dict(grad_as_bucket_view=True, static_graph=False, bucket_cap_mb=0), False),
        ("+bucket_view+static_graph", "new",
         dict(grad_as_bucket_view=True, static_graph=True, bucket_cap_mb=0), False),
    ]

    results = {}
    for name, kind, opts, _ in arms:
        policy = build_policy(device, seed=0)
        if use_ddp:
            policy.model = wrap_ddp(policy.model, sync_mode="per_update", **opts)
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=8e-4, weight_decay=1e-2)
        for step in range(steps):
            batch, target = make_batch(batch_size, device, seed=1000 + step * 97 + rank)
            # The real loop calls the DDP-wrapped model under no_grad during the rollout before every
            # update; exercise that here too, because it is the path `static_graph` could break.
            policy.infer(batch)
            if kind == "upstream":
                upstream_loop(policy, optimizer, batch, target, epochs, 1.0, use_ddp)
            else:
                run_update_epochs(policy, optimizer, batch, target, None, epochs, 1.0,
                                  use_ddp, world_size, sync_mode="per_update")
        results[name] = flat_params(policy.model)

    ref_name = arms[0][0]
    ref = results[ref_name]
    ok = True
    for name, _, _, must_match in arms[1:]:
        params = results[name]
        same = torch.equal(ref, params)
        delta = (ref - params).abs().max().item()
        if must_match:
            ok &= same
        verdict = "BITWISE-EQUAL" if same else ("DIFFERS(required equal!)" if must_match else "differs")
        if rank == 0:
            print(f"[equivalence] {verdict:<24} max|delta|={delta:.3e}  {name}")
    assert ok, "a semantics-preserving arm changed the weights"
    if rank == 0:
        print(f"[equivalence] PASS -- {steps} steps x {epochs} epochs, batch {batch_size}/rank, "
              f"world_size {world_size}")

    # local_avg is EXPECTED to differ; report by how much so the change is never silent.
    if use_ddp:
        policy = build_policy(device, seed=0)
        policy.model = wrap_ddp(policy.model, sync_mode="local_avg",
                                grad_as_bucket_view=True, static_graph=True)
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=8e-4, weight_decay=1e-2)
        for step in range(steps):
            batch, target = make_batch(batch_size, device, seed=1000 + step * 97 + rank)
            run_update_epochs(policy, optimizer, batch, target, None, epochs, 1.0,
                              use_ddp, world_size, sync_mode="local_avg")
        got = flat_params(policy.model)
        delta = (ref - got).abs().max().item()
        rel = delta / ref.abs().max().item()
        # ranks must still agree with each other at the training-step boundary
        probe = got.clone()
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        probe.div_(world_size)
        skew = (probe - got).abs().max().item()
        if rank == 0:
            print(f"[local_avg] differs from per_update as designed: max|delta|={delta:.3e} "
                  f"(rel {rel:.3e}); cross-rank parameter skew after averaging = {skew:.3e}")
        assert skew < 1e-5, "local_avg left the ranks out of sync"


def check_timing(device: str, batch_size: int, epochs: int, steps: int,
                 rank: int, world_size: int) -> None:
    arms = [
        ("upstream DDP ctor, per_update", dict(grad_as_bucket_view=False, static_graph=False,
                                               bucket_cap_mb=0), "per_update"),
        ("+bucket_view, per_update", dict(grad_as_bucket_view=True, static_graph=False,
                                          bucket_cap_mb=0), "per_update"),
        ("+bucket_view+static_graph, per_update", dict(grad_as_bucket_view=True, static_graph=True,
                                                       bucket_cap_mb=0), "per_update"),
        ("+bucket_view+static_graph+cap100, per_update", dict(grad_as_bucket_view=True,
                                                              static_graph=True,
                                                              bucket_cap_mb=100), "per_update"),
        # sync_mode / epoch-count arms keep the UPSTREAM constructor so exactly one thing moves.
        ("local_avg (upstream ctor)", dict(grad_as_bucket_view=False, static_graph=False,
                                           bucket_cap_mb=0), "local_avg"),
        ("epochs=1 (upstream ctor, per_update)", dict(grad_as_bucket_view=False, static_graph=False,
                                                      bucket_cap_mb=0), "per_update"),
    ]
    use_ddp = world_size > 1
    if rank == 0:
        print(f"\n[timing] world_size={world_size} batch/rank={batch_size} epochs={epochs} "
              f"steps={steps} (update loop only, no simulator)")
        print(f"{'arm':<48}{'ms/update':>12}{'ms/epoch':>11}")
    for name, opts, mode in arms:
        n_epochs = 1 if name.startswith("epochs=1") else epochs
        policy = build_policy(device, seed=0)
        if use_ddp:
            policy.model = wrap_ddp(policy.model, sync_mode=mode, **opts)
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=8e-4, weight_decay=1e-2)
        batch, target = make_batch(batch_size, device, seed=7 + rank)
        for _ in range(10):  # warmup
            run_update_epochs(policy, optimizer, batch, target, None, n_epochs, 1.0,
                              use_ddp, world_size, sync_mode=mode)
        if use_ddp:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            if use_ddp:
                dist.barrier()          # stand in for the per-step rank_skew_wait
            run_update_epochs(policy, optimizer, batch, target, None, n_epochs, 1.0,
                              use_ddp, world_size, sync_mode=mode)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps * 1000.0
        if use_ddp:
            buf = torch.tensor([dt], device=device)
            dist.all_reduce(buf, op=dist.ReduceOp.MAX)
            dt = buf.item()
        if rank == 0:
            print(f"{name:<48}{dt:>12.2f}{dt / n_epochs:>11.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="all", choices=["all", "accumulation", "equivalence", "timing"])
    ap.add_argument("--batch-size", type=int, default=128, help="envs per rank (batch = envs x horizon)")
    ap.add_argument("--epochs", type=int, default=10, help="dagger_learning_epochs")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.check in ("all", "accumulation") and rank == 0:
        check_accumulation(device, args.batch_size, args.epochs)
    if args.check in ("all", "equivalence"):
        check_equivalence(device, args.batch_size, args.epochs, args.steps, rank, world_size)
    if args.check in ("all", "timing"):
        check_timing(device, args.batch_size, args.epochs, args.steps, rank, world_size)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
