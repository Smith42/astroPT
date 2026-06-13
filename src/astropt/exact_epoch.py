"""
Exact one-epoch data loading for AstroPT.

Streaming a HF dataset with `split_dataset_by_node` + a buffered shuffle gives an
example order that depends on the GPU count and the DataLoader worker count, and it
cannot guarantee "every example exactly once" -- the standard distributed samplers
either drop the tail or pad it with duplicates when ``len(dataset) % world_size != 0``.

For a scaling study where each run must train on the IDENTICAL data, exactly once,
we instead use a local map-style dataset and the sampler below.

``ExactDistributedSampler`` partitions ``range(dataset_len)`` across ranks with NO
padding and NO dropping: a single shared shuffle (seeded only by ``seed``) is sliced
strided, so rank ``r`` owns ``indices[r::world_size]``. Properties:

* **Exact** -- the union of all ranks is exactly ``{0, ..., dataset_len-1}``; per-rank
  counts differ by at most one, so one epoch sees every example exactly once.
* **Reproducible** -- the order depends only on ``seed`` (and ``dataset_len``), so every
  config that shares ``seed`` trains on the identical global order.
* **Worker-independent** -- the sampler fixes the index order; DataLoader workers only
  fetch in parallel, they do not reorder. So ``num_workers`` no longer affects the order.

With micro-batch ``b`` and per-rank accumulation ``a``, every rank runs exactly
``ceil(ceil(n_r / b) / a)`` optimizer steps. For the AstroPT galaxies set
(N = 8,474,566) at b=16, total grad-accum 40 and world sizes {1,2,4,8}, that is 13242
steps on *every* rank -- balanced, so DDP needs no Join; only the final accumulation
step is short. See ``planned_steps`` / the ``__main__`` self-test below.
"""

import math

import numpy as np
import torch
from torch.utils.data import Sampler


class ExactDistributedSampler(Sampler):
    """Strided, padding-free, drop-free partition of ``range(dataset_len)`` across ranks.

    Args:
        dataset_len: number of examples in the (map-style) dataset.
        num_replicas: world size (number of DDP ranks).
        rank: this process's rank in ``[0, num_replicas)``.
        seed: shuffle seed -- keep it fixed across configs for identical ordering.
        shuffle: shuffle before partitioning (True for training).
    """

    def __init__(self, dataset_len, num_replicas, rank, seed=1337, shuffle=True):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for num_replicas {num_replicas}")
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

    def _indices(self):
        idx = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(idx)
        return idx[self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(self._indices().tolist())

    def __len__(self):
        # number of indices i in [0, dataset_len) with i % num_replicas == rank
        return len(range(self.rank, self.dataset_len, self.num_replicas))


def planned_steps(dataset_len, num_replicas, micro_batch, grad_accum):
    """Optimizer steps each rank will run for one exact epoch, as a list (one per rank).

    ``grad_accum`` is the GLOBAL accumulation (it is split across ranks). All entries are
    equal when the partition balances; if they are NOT all equal, plain DDP would hang and
    you would need the Join context manager -- the training loop should assert on this.
    """
    if grad_accum % num_replicas != 0:
        raise ValueError(f"grad_accum {grad_accum} not divisible by num_replicas {num_replicas}")
    per_rank_accum = grad_accum // num_replicas
    steps = []
    for r in range(num_replicas):
        n_r = len(range(r, dataset_len, num_replicas))
        micro = math.ceil(n_r / micro_batch)
        steps.append(math.ceil(micro / per_rank_accum))
    return steps


def one_epoch_loop(
    *,
    model,
    loader,
    optimizer,
    scaler,
    ctx,
    process_batch,
    grad_accum_per_rank,
    grad_clip,
    get_lr,
    count_examples,
    log_interval=100,
    eval_interval=None,
    checkpoint_iters=(),
    on_eval=None,
    on_checkpoint=None,
    master=True,
    start_iter=0,
    log=print,
):
    """Run exactly one pass over ``loader`` (which must use an ``ExactDistributedSampler``).

    An optimizer step is taken every ``grad_accum_per_rank`` micro-batches; the final step
    of the epoch is short (fewer micro-batches) but -- because the exact partition gives
    every rank the SAME micro-batch count -- it is short by the same amount on every rank,
    so the per-step gradient all-reduce count stays balanced and DDP needs no Join.

    Gradients are synced on every backward (no ``no_sync``): all-reduce is linear, so this
    is numerically identical to accumulate-then-sync, and it keeps the short final step
    correct. Returns ``(iter_num, examples_seen)``; assert the all-reduced ``examples_seen``
    equals ``len(dataset)`` to prove exact coverage.
    """
    it = iter(loader)
    for _ in range(start_iter * grad_accum_per_rank):  # resume: skip consumed micro-batches
        if next(it, None) is None:
            break
    iter_num = start_iter
    seen = 0
    last_loss = float("nan")
    while True:
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        if master and on_eval and eval_interval and iter_num % eval_interval == 0:
            on_eval(iter_num)
        if master and on_checkpoint and iter_num in checkpoint_iters:
            on_checkpoint(iter_num)
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        for _ in range(grad_accum_per_rank):
            try:
                raw = next(it)
            except StopIteration:
                break
            B = process_batch(raw)
            seen += count_examples(B)
            with ctx:
                _, loss = model(B["X"], targets=B["Y"])
            scaler.scale(loss / grad_accum_per_rank).backward()
            last_loss = loss.item()
            micro += 1
        if micro == 0:  # loader exhausted exactly at a step boundary -> epoch done
            break
        if grad_clip and grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        iter_num += 1
        if master and log_interval and iter_num % log_interval == 0:
            log(f"iter {iter_num}: loss {last_loss * grad_accum_per_rank:.4f} lr {lr:.2e}")
    return iter_num, seen


if __name__ == "__main__":
    # Self-test: the partition is exact (covers every index once) and step-balanced.
    def _check(N, W, b, A):
        seen = []
        for r in range(W):
            seen.extend(ExactDistributedSampler(N, W, r, seed=1337)._indices().tolist())
        assert sorted(seen) == list(range(N)), f"partition not exact for N={N} W={W}"
        steps = planned_steps(N, W, b, A)
        balanced = len(set(steps)) == 1
        print(f"N={N:>9} W={W} b={b} A={A}: exact=True steps={steps[0]} balanced={balanced}")
        return balanced

    ok = True
    for W in (1, 2, 4, 8):
        ok &= _check(8_474_566, W, 16, 40)  # the AstroPT galaxies train split
    for N in (1000, 1001, 1024, 999_983):   # assorted sizes incl. a prime
        for W in (1, 2, 3, 4):
            seen = []
            for r in range(W):
                seen.extend(ExactDistributedSampler(N, W, r)._indices().tolist())
            assert sorted(seen) == list(range(N)), f"partition not exact for N={N} W={W}"
    print("exact partition verified for all cases; galaxies set step-balanced:", ok)

    # CPU control-flow test of one_epoch_loop: it must consume EXACTLY N examples and run
    # the expected number of optimizer steps, including a short final accumulation step.
    from contextlib import nullcontext

    from torch.utils.data import DataLoader, Dataset

    class _Toy(Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return torch.tensor([float(i)])

    class _ToyModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
        def forward(self, X, targets=None): return None, (self.w * X).sum()

    for N, b, accpr in [(1000, 4, 3), (8474566 // 4, 16, 10)]:  # 2nd = one W=4 rank slice
        ds = _Toy(N)
        loader = DataLoader(ds, batch_size=b,
                            sampler=ExactDistributedSampler(N, 1, 0, seed=1337),
                            drop_last=False)
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        steps, seen = one_epoch_loop(
            model=model, loader=loader, optimizer=opt,
            scaler=torch.amp.GradScaler("cpu", enabled=False), ctx=nullcontext(),
            process_batch=lambda raw: {"X": raw, "Y": raw},
            grad_accum_per_rank=accpr, grad_clip=0.0, get_lr=lambda it: 0.0,
            count_examples=lambda B: B["X"].shape[0], log_interval=0, master=False,
        )
        expected_steps = math.ceil(math.ceil(N / b) / accpr)
        assert seen == N, f"coverage {seen} != {N}"
        assert steps == expected_steps, f"steps {steps} != {expected_steps}"
        print(f"one_epoch_loop: N={N:>9} b={b} accpr={accpr} -> seen={seen} steps={steps} (exact)")
    print("one_epoch_loop consumes exactly N with the expected step count.")
