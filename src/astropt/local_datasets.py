"""
A place to store our pytorch datasets.
"""

import os
import random

import einops
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset
from torchvision import io


class GalaxyImageDataset(Dataset):
    def __init__(
        self,
        paths={"images": None, "spectra": None},
        transform=None,
        stochastic=True,
        spiral=False,
        modality_registry=None,
    ):
        """
        Arguments:
            paths (callable, optional): files with all the galaxy paths. Paths can be None if streaming from HF.
            transform (callable, optional): Optional transform to be applied on a sample.
            stochastic: sample at random
            spiral: spiral form instead of raster form
            modality_registry: size of ViT patch
        """
        if paths is not None:
            self.paths = {}
            for modality, path in paths.items():
                self.paths[modality] = None
                if paths[modality] is not None:
                    self.paths[modality] = np.genfromtxt(path, delimiter=",", dtype=str)
                else:
                    self.paths[modality] = None
            # get length of first dataset that is not None
            try:
                self.dataset_len = len(
                    next(v for v in self.paths.values() if v is not None)
                )
            except StopIteration:
                self.dataset_len = None
        else:
            self.paths = paths
        self.transform = transform
        self.modality_registry = modality_registry
        self.stochastic = stochastic
        self.spiral = spiral

    def __len__(self):
        # set to a big number if stochastic as a fudge for epochism in pytorch
        return int(1e10) if self.stochastic else self.dataset_len

    @staticmethod
    def _spiral(n):
        """
        generate a spiral index array of side length 'n'
        there must be a better way to do this: any suggestions?
        """
        a = np.arange(n * n)
        b = a.reshape((n, n))
        m = None
        for i in range(n, 0, -2):
            m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
            b = b[1:-1, 1:-1]
        a[list(m[1:])] = list(a)
        a = abs(a - n * n + 1)
        return a.reshape((n, n))

    def spiralise(self, galaxy):
        """
        Change ViT patch ordering to a 'spiral order'. See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        Alternate function available here:
        https://www.procook.co.uk/product/procook-spiralizer-black-and-stainless-steel
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        spiraled = [ii for _, ii in sorted(zip(indices, galaxy))]
        return (
            torch.stack(spiraled)
            if isinstance(spiraled[0], torch.Tensor)
            else np.stack(spiraled)
        )

    def antispiralise(self, galaxy):
        """
        Change ViT patch ordering from spiral to raster order. See 'spiralise'.
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        antispiraled = [galaxy[ii] for ii in indices]
        return (
            torch.stack(antispiraled)
            if isinstance(antispiraled[0], torch.Tensor)
            else np.stack(antispiraled)
        )

    def process_galaxy(self, raw_galaxy):
        patch_size = self.modality_registry.get_config("images").patch_size
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        if "images" in self.transform:
            patch_galaxy = self.transform["images"](patch_galaxy)
        if self.spiral:
            patch_galaxy = self.spiralise(patch_galaxy)

        return patch_galaxy

    def process_spectra(self, raw_spectra, wavelength):
        patch_size = self.modality_registry.get_config("spectra").patch_size
        # Apply padding to the spectrum
        w = raw_spectra.shape[0]
        window = patch_size
        pad_w = (window - w % window) % window
        padded_spectra = F.pad(raw_spectra, (0, pad_w))
        padded_wl = F.pad(wavelength, (0, pad_w))

        # Now rearrange into patches
        patch_spectra = einops.rearrange(
            padded_spectra,
            "(w p) -> (w) (p)",
            p=window,
        )

        patch_wl = einops.rearrange(
            padded_wl,
            "(w p) -> (w) (p)",
            p=window,
        )

        if "spectra" in self.transform:
            patch_spectra = self.transform["spectra"](patch_spectra)

        return patch_spectra, patch_wl

    @staticmethod
    def process_modes(x, modality_registry, device, shuf=False):
        """Move all tensor values in dictionary x to the specified device.
        And split into X and Y according to the modality registry."""
        modes = modality_registry.generate_sequence(shuf=shuf)

        # Move all tensors to device first
        x_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()
        }

        X = {}
        Y = {}
        for ii, mode in enumerate(modes):
            X[mode] = x_on_device[mode]
            X[f"{mode}_positions"] = x_on_device[f"{mode}_positions"]
            Y[mode] = x_on_device[mode]
            if ii == 0:
                Y[mode] = Y[mode][:, 1:]
            if len(modes) == 1:
                X[mode] = X[mode][:, :-1]
                X[f"{mode}_positions"] = X[f"{mode}_positions"][:, :-1]

        return {"X": X, "Y": Y}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.stochastic:
            idx = np.random.randint(self.dataset_len)

        while True:
            # be careful in this loop -- it fails quietly if there is an error!!!
            try:
                if "images" in self.paths:
                    # get extension to filename and read via FITS or JPG
                    _, ext = (
                        os.path.splitext(self.paths["images"][idx])
                        if len(self.paths["images"].shape) == 1
                        else os.path.splitext(self.paths["images"][idx][0])
                    )
                    if ext == ".jpg" or ext == ".jpeg":
                        raw_galaxy = io.read_image(str(self.paths["images"][idx])).to(
                            torch.bfloat16
                        )
                    elif ext == ".fits" or ext == ".FITS":
                        if len(self.paths["images"].shape) == 1:
                            # case with 1 FITS file
                            with fits.open(self.paths["images"][idx]) as hdul:
                                raw_galaxy = hdul[0].data.astype(
                                    np.float32
                                )  # Assuming the image data is in the first HDU
                                # we need to convert to float32 as FITS has bigendian issues (https://stackoverflow.com/questions/59247385/why-does-torch-from-numpy-require-a-different-byte-ordering-while-matplotlib-doe)
                            raw_galaxy = torch.tensor(raw_galaxy[np.newaxis]).to(
                                torch.bfloat16
                            )
                        else:
                            # case with N FITS files
                            raw_galaxy = []
                            for path in self.paths["images"][idx]:
                                with fits.open(path) as hdul:
                                    raw_galaxy.append(
                                        hdul[0].data.astype(np.float32)
                                    )  # Assuming the image data is in the first HDU
                                    # we need to convert to float32 as FITS has bigendian issues
                            raw_galaxy = torch.tensor(np.stack(raw_galaxy)).to(
                                torch.bfloat16
                            )
                    else:
                        raise NotImplementedError(
                            f"File must be FITS or JPEG, it is instead {ext}."
                        )
                    patch_galaxy = self.process_galaxy(raw_galaxy)
                    if torch.isnan(patch_galaxy).any():
                        raise ValueError("Found NaNs in galaxy, skipping file")
                else:
                    patch_galaxy = np.array([np.nan])
                if "spectra" in self.paths:
                    # Fetch spectrum if we have one
                    with fits.open(self.paths["spectra"][idx]) as hdul:
                        raw_spectra = hdul[1].data["Flux"].astype(np.float32)
                        wavelength = hdul[1].data["Wave"].astype(np.float32)
                    raw_spectra = torch.tensor(raw_spectra).to(torch.bfloat16)
                    wavelength = (
                        torch.tensor(wavelength).to(torch.bfloat16) - 3000
                    ) / (10000 - 3000)
                    patch_spectra, patch_wl = self.process_spectra(
                        raw_spectra, wavelength
                    )

                    if torch.isnan(patch_spectra).any() or torch.isnan(patch_wl).any():
                        raise ValueError("Found NaNs in spectra, skipping file")
                else:
                    patch_spectra = np.array([np.nan])
                    patch_wl = np.array([np.nan])
                break
            except Exception as err:
                if self.stochastic:
                    idx = np.random.randint(len(self.paths))
                else:
                    raise err

        return {
            "images": patch_galaxy,
            "images_positions": torch.arange(0, len(patch_galaxy), dtype=torch.long),
            "spectra": patch_spectra,
            "spectra_positions": patch_wl,
            "idx": idx,
        }


class PhotometryDataset(Dataset):
    """
    Dataset for photometry timeseries packaging for AstroPT.

    Args:
        split: Dataset split from HF dataset
    """

    def __init__(self, split, roll=False, shuffle=False):
        self.split = split
        self.shuffle = shuffle  # whether to roll time
        self.roll = roll  # whether to roll time

    def __len__(self):
        return len(self.split)

    @staticmethod
    def embed_time(time):
        embedded_time = torch.stack(
            (torch.sin(2 * torch.pi * time), torch.cos(2 * torch.pi * time)), dim=-1
        )
        return embedded_time

    @staticmethod
    def collate_fn(batch):
        """
        Expects batch items to have structure:
        {
            'photometry': tensor/list,
            'photometry_positions': tensor/list,
        }
        """
        photometry_data = [item["photometry"] for item in batch]
        photometry_positions = [item["photometry_positions"] for item in batch]

        padded_photometry = pad_sequence(
            photometry_data, batch_first=True, padding_value=0.0
        )
        padded_positions = pad_sequence(
            photometry_positions, batch_first=True, padding_value=0.0
        )

        lengths = [len(seq) for seq in photometry_data]
        max_len = padded_photometry.size(1)
        attention_mask = torch.zeros(len(batch), max_len)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1

        return {
            "photometry": padded_photometry,
            "photometry_positions": padded_positions,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx):
        item = self.split[idx]["photometry"]
        if self.shuffle:
            indices = torch.randperm(len(item))
            item = item[indices]
        flux_et_al = item[:-1, 1:2]
        # As photometry is stochastically sampled we feed the model the current
        # timestamp and next timestamp in the sequence as the position vector
        time = self.embed_time(item[:, 0])
        time = torch.cat((time[1:], time[:-1]), dim=1)
        if self.roll:
            roll_by = np.random.randint(len(flux_et_al))
            return {
                "photometry": flux_et_al.roll(roll_by, 0),
                "photometry_positions": time.roll(roll_by, 0),
                "idx": idx,
            }
        else:
            return {
                "photometry": flux_et_al,
                "photometry_positions": time,
                "idx": idx,
            }


class SpectraDataset(Dataset):
    """
    Dataset for AstroM3 spectra packaging for AstroPT.

    Args:
        split: Dataset split from HF dataset
    """

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[idx]["spectra"]
        flux_et_al = item[1:2, ::2].T
        wavelength = torch.arange(0, len(flux_et_al), dtype=torch.long)
        # wavelength = item[0:1, ::2].T
        return {
            "spectra": flux_et_al,
            "spectra_positions": wavelength,
            "idx": idx,
        }


class MetadataDataset(Dataset):
    """
    Dataset for AstroM3 metadata packaging for AstroPT.

    Args:
        split: Dataset split from HF dataset
    """

    def __init__(self, split, shuffle_time=False):
        self.split = split
        self.shuffle = shuffle_time

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[idx]["metadata"]
        metadata = item.unsqueeze(-1)
        metadata_positions = torch.arange(0, len(metadata), dtype=torch.long)
        if self.shuffle:
            perm = torch.randperm(metadata.size(0))
            metadata = metadata[perm]
            metadata_positions = metadata_positions[perm]
        return {
            "metadata": metadata,
            "metadata_positions": metadata_positions,
            "idx": idx,
        }


class LLMModalityDataset(IterableDataset):
    """
    Dataset that creates LLM-compatible sequences with special tokens for modalities.
    Raw modality data is preserved for model encoding.
    """

    def __init__(
        self,
        hf_dataset,
        modality_registry,
        tokenizer,
        special_token_ids,
        transforms=None,
        random_order=True,
        apply_chat_template=True,
    ):
        """
        Args:
            hf_dataset: Hugging Face dataset
            modality_registry: ModalityRegistry instance
            tokenizer: HF tokenizer with special tokens
            special_token_ids: Dict mapping special token strings to IDs
            transforms: Optional transforms dict
            random_order: Whether to randomize modality order per sample
            apply_chat_template: Apply chat template
            text_prompt: Optional text prompt to prepend
        """
        self.dataset = hf_dataset
        self.modality_registry = modality_registry
        self.tokenizer = tokenizer
        self.special_token_ids = special_token_ids
        self.transforms = transforms if transforms is not None else {}
        self.random_order = random_order
        self.apply_chat_template = apply_chat_template

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _spiral(n):
        """
        generate a spiral index array of side length 'n'
        there must be a better way to do this: any suggestions?
        """
        a = np.arange(n * n)
        b = a.reshape((n, n))
        m = None
        for i in range(n, 0, -2):
            m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
            b = b[1:-1, 1:-1]
        a[list(m[1:])] = list(a)
        a = abs(a - n * n + 1)
        return a.reshape((n, n))

    def spiralise(self, galaxy):
        """
        Change ViT patch ordering to a 'spiral order'. See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        Alternate function available here:
        https://www.procook.co.uk/product/procook-spiralizer-black-and-stainless-steel
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        spiraled = [ii for _, ii in sorted(zip(indices, galaxy))]
        return (
            torch.stack(spiraled)
            if isinstance(spiraled[0], torch.Tensor)
            else np.stack(spiraled)
        )

    def antispiralise(self, galaxy):
        """
        Change ViT patch ordering from spiral to raster order. See 'spiralise'.
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        antispiraled = [galaxy[ii] for ii in indices]
        return (
            torch.stack(antispiraled)
            if isinstance(antispiraled[0], torch.Tensor)
            else np.stack(antispiraled)
        )

    def process_galaxy(self, galaxy):
        """Process galaxy image into patches"""
        patch_size = self.modality_registry.get_config("images").patch_size
        galaxy = einops.rearrange(
            galaxy,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        if "images" in self.transforms:
            galaxy = self.transforms["images"](galaxy)
        galaxy = self.spiralise(galaxy)

        return galaxy

    def create_sequence_structure(self, sample_data, image_first=True):
        """
        Create sequence structure with special tokens.
        Returns token sequence and modality metadata (no encoding).
        """
        sequence_parts = []
        modality_info = {
            "names": [],
            "starts": [],
            "lengths": [],
            "data": [],
            "positions": [],
        }
        current_pos = 0

        # Get modality order (random or fixed)
        if self.random_order:
            modality_order = random.sample(
                self.modality_registry.names(), len(self.modality_registry.names())
            )
        else:
            modality_order = self.modality_registry.names()

        if image_first:
            modality_order.remove("images")
            modality_order.insert(0, "images")

        ii = 0

        if self.apply_chat_template:
            sys_prompt = self.tokenizer.encode(
                "<|im_start|>system\nYou are a helpful AI assistant named AstroPetey, finetuned from SmolLM for astronomical applications.\n\n<|im_end|>\n"
            )
            sequence_parts.extend(sys_prompt)
            current_pos += len(sys_prompt)

        for mod_name in modality_order:
            if mod_name in sample_data:
                if self.apply_chat_template:
                    if ii == 0:
                        use_prompt = self.tokenizer.encode("<|im_start|>user\n")
                        sequence_parts.extend(use_prompt)
                        current_pos += len(use_prompt)
                    if ii == 1:
                        ass_prompt = self.tokenizer.encode("<|im_start|>assistant\n")
                        sequence_parts.extend(ass_prompt)
                        current_pos += len(ass_prompt)

                begin_token = self.tokenizer.encode(f"<|begin_{mod_name}|>")
                sequence_parts.extend(begin_token)
                current_pos += 1

                mod_data = sample_data[mod_name]
                mod_pos = sample_data[mod_name + "_positions"]
                seq_len = mod_data.shape[0]  # Number of patches/tokens

                modality_info["names"].append(mod_name)
                modality_info["starts"].append(current_pos)
                modality_info["lengths"].append(seq_len)
                modality_info["data"].append(mod_data)
                modality_info["positions"].append(mod_pos)

                # Add placeholder tokens for modality data
                placeholder_tokens = self.tokenizer.encode(f"<|{mod_name}|>")
                for _ in range(seq_len):
                    sequence_parts.extend(placeholder_tokens)
                current_pos += seq_len

                end_token = self.tokenizer.encode(f"<|end_{mod_name}|>")
                sequence_parts.extend(end_token)
                current_pos += 1

                if self.apply_chat_template and ii == 0:
                    prompt = self.tokenizer.encode("<|im_end|>\n")
                    sequence_parts.extend(prompt)
                    current_pos += len(prompt)

                ii += 1

        if self.apply_chat_template:
            prompt = self.tokenizer.encode("<|im_end|>\n")
            sequence_parts.extend(prompt)
            current_pos += len(prompt)

        token_sequence = torch.tensor(sequence_parts, dtype=torch.long)
        attention_mask = torch.ones_like(token_sequence)

        return token_sequence, attention_mask, modality_info

    def __iter__(self):
        """Get a single sample with sequence structure"""
        for raw_sample in self.dataset:
            sample_data = {}
            for key in raw_sample.keys():
                if (
                    (key not in ["dr8_id", "image"])
                    and (raw_sample[key] is not None)
                    and (raw_sample[key] > -90)
                    and np.isfinite(raw_sample[key])
                ):
                    # assume all other inputs that aren't image crop or dr8_id
                    # are parameters
                    sample_data[key] = torch.tensor([raw_sample[key]])
                    sample_data[f"{key}_positions"] = torch.tensor(
                        [0], dtype=torch.long
                    )

            if "image" in raw_sample:
                galaxy = torch.from_numpy(
                    np.array(raw_sample["image"]).swapaxes(0, 2)
                ).to(torch.float)
                galaxy = self.process_galaxy(galaxy)

                sample_data["images"] = galaxy
                sample_data["images_positions"] = torch.arange(
                    0, len(galaxy), dtype=torch.long
                )

            token_sequence, attention_mask, modality_info = (
                self.create_sequence_structure(sample_data)
            )

            yield {
                "token_sequence": token_sequence,
                "attention_mask": attention_mask,
                "modality_info": modality_info,
            }


def llm_collate_fn(batch):
    """
    Collate function that pads sequences and groups modality info.
    """

    def _target_modality_info(modality_info):
        """Update modality info for target array"""
        result = modality_info.copy()
        result["starts"] = [start - 1 for start in modality_info["starts"]]
        return result

    token_sequences = [item["token_sequence"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    modality_infos = [item["modality_info"] for item in batch]

    # Pad sequences to same length
    padded_tokens = pad_sequence(token_sequences, batch_first=True, padding_value=0)
    padded_attention = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    X = {
        "token_sequences": padded_tokens[:, :-1],
        "attention_masks": padded_attention[:, :-1],
        "modality_infos": modality_infos,
    }
    Y = {
        "token_sequences": padded_tokens[:, 1:],
        "attention_masks": padded_attention[:, 1:],
        "modality_infos": [_target_modality_info(info) for info in modality_infos],
    }

    return {"X": X, "Y": Y}
