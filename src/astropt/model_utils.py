import gc

import torch
from huggingface_hub import hf_hub_download


def _synth_batch_like(ref, batch_size):
    """Build a synthetic X (or Y) dict of the given batch size from a real one.

    Float tensors (patch content / targets) become random tensors of the same
    trailing shape and dtype; integer tensors (e.g. *_positions) are tiled from
    a real row so they stay valid indices for the embedders.
    """
    out = {}
    for k, v in ref.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        shape = (batch_size,) + tuple(v.shape[1:])
        if v.dtype in (torch.int64, torch.int32):
            out[k] = v[0:1].expand(shape).contiguous()
        else:
            out[k] = torch.randn(shape, dtype=v.dtype, device=v.device)
    return out


def find_max_batch_size(
    model,
    sample,
    device,
    ctx,
    *,
    start=1,
    max_cap=2048,
    safety=0.9,
    ddp=False,
    master_process=True,
):
    """Probe the largest micro-batch that fits in GPU memory via a real fwd+bwd.

    Doubles the batch size until a CUDA out-of-memory error, then returns
    ``int(safety * largest_that_fit)`` to leave headroom for evaluation and
    allocator fragmentation.

    A throwaway AdamW optimizer with lr=0 is stepped during probing so that
    optimizer-state memory is included in the peak, while leaving the model
    weights unchanged (an lr=0 step is a no-op update). Probing does no
    collective ops, so under DDP each rank measures independently and then a
    single all_reduce(MIN) makes every rank agree on the smallest fitting size
    (mismatched micro-batches across ranks would deadlock the all-reduce).

    Args:
        model: the (uncompiled, un-DDP-wrapped) model, already on ``device``.
        sample: a {"X": ..., "Y": ...} dict from ``process_modes`` used only to
            learn tensor shapes/dtypes.
        ctx: the autocast context manager used for training.

    Returns:
        The chosen micro-batch size (int, >= 1).
    """
    X0, Y0 = sample["X"], sample["Y"]
    probe_opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    was_training = model.training
    model.train()
    if master_process:
        print("auto_find_batch_size: probing largest micro-batch that fits...")

    best = 0
    batch = max(1, start)
    while batch <= max_cap:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            X = _synth_batch_like(X0, batch)
            Y = _synth_batch_like(Y0, batch)
            probe_opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(X, targets=Y)
            loss.backward()
            probe_opt.step()
            best = batch
            if master_process:
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"  micro-batch {batch}: fits ({peak:.2f} GB peak)")
            del X, Y, loss
            batch *= 2
        except Exception as err:  # noqa: BLE001 - re-raised unless it is an OOM
            if "out of memory" not in str(err).lower():
                raise
            if master_process:
                print(f"  micro-batch {batch}: OOM")
            break

    # tidy up so probe state does not leak into real training. gc.collect()
    # before empty_cache() is needed: reference cycles in the autograd graph /
    # optimizer otherwise keep the (already-freed) blocks reserved, leaving the
    # card looking full to the rest of the run.
    model.zero_grad(set_to_none=True)
    del probe_opt
    gc.collect()
    torch.cuda.empty_cache()
    if not was_training:
        model.eval()

    chosen = max(1, int(best * safety))
    if ddp:
        import torch.distributed as dist

        agreed = torch.tensor([chosen], device=device)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        chosen = int(agreed.item())
    if master_process:
        print(
            f"auto_find_batch_size: using micro-batch {chosen} "
            f"(largest fit {best}, safety {safety})"
        )
    return chosen


def load_astropt(
    repo_id="smith42/astropt_sparse",
    path="astropt/p16k10",
    weights_filename="ckpt.pt",
    use_llm_backbone=False,
    llm_model_name=None,
):
    """
    Load an AstroPT model.

    Args:
        repo_id: The Hugging Face repo ID (e.g., "smith42/astropt_sparse"). If this is None we assume that we are loading a local model.
        path: Subdirectory containing the model (e.g., "sparse" or "dense")
        weights_filename: Name of the weights file

    Returns:
        Loaded model
    """
    from astropt.model import GPT, GPTConfig

    if repo_id is not None:
        # Ping root config.json to count as a download at the repo level
        try:
            _ = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",  # Root config.json
                local_files_only=False,
            )
        except Exception as e:
            print(e)
            pass

        if path:
            weights_filepath = f"{path}/{weights_filename}"
        else:
            weights_filepath = weights_filename

        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filepath,
            force_download=False,
        )
    else:
        weights_path = path + weights_filename

    checkpoint = torch.load(weights_path, weights_only=False, map_location="cpu")
    model_args = checkpoint["model_args"]
    modality_registry = checkpoint["modality_registry"]

    config = GPTConfig(**model_args)
    model = GPT(
        config,
        modality_registry,
    )

    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # torch.compile adds _orig_mod. prefix to parameter names so we fix below
    unwanted_prefix = "_orig_mod."
    keys_to_update = []
    for k in state_dict.keys():
        if unwanted_prefix in k:
            new_key = k.replace(unwanted_prefix, "")
            keys_to_update.append((k, new_key))
    for old_key, new_key in keys_to_update:
        state_dict[new_key] = state_dict.pop(old_key)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        print(err)
        print("Assuming we are loading a version of AstroPT < 3.0, so altering state dict key names to fit...")
        state_dict["encoders.images.c_fc.weight"] = state_dict.pop("transformer.wte.images.c_fc.weight")
        state_dict["encoders.images.c_proj.weight"] = state_dict.pop("transformer.wte.images.c_proj.weight")
        state_dict["decoders.images.c_fc.weight"] = state_dict.pop("lm_head.images.c_fc.weight")
        state_dict["decoders.images.c_proj.weight"] = state_dict.pop("lm_head.images.c_proj.weight")
        state_dict["embedders.images.wpe.weight"] = state_dict.pop("transformer.wpe.images.wpe.weight")
        model.load_state_dict(state_dict)


    dir_info = f"/{path}" if path else ""
    print(f"model loaded successfully from {repo_id}{dir_info}")
    print("args:", model_args)
    return model
