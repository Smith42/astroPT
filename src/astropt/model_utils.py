import torch
from huggingface_hub import hf_hub_download


def load_astropt(
    repo_id="smith42/astropt_sparse",
    path="astropt/p16k10",
    weights_filename="ckpt.pt",
):
    """
    Load an AstroPT model.

    Args:
        repo_id: The Hugging Face repo ID (e.g., "smith42/astropt_sparse")
        path: Subdirectory containing the model (e.g., "sparse" or "dense")
        weights_filename: Name of the weights file

    Returns:
        Loaded model
    """
    from astropt.model import GPT, GPTConfig

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
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    dir_info = f"/{path}" if path else ""
    print(f"model loaded successfully from {repo_id}{dir_info}")
    print("args:", model_args)
    return model
