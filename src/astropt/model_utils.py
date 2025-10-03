import torch
from huggingface_hub import hf_hub_download


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
