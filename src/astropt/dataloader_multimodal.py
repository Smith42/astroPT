import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets, load_dataset
from torchvision import transforms
from typing import Dict, Any, List, Optional

from astropt.processors import ModalityProcessor

class MultimodalDatasetArrow(Dataset):
    """
    Orchestrator dataset that handles multiple modalities in a modular way.
    """
    def __init__(
        self,
        arrow_folder_root: str,
        split: str,
        processors: Dict[str, ModalityProcessor],
        modality_registry: Any,
        transform: Dict[str, Any] = {},
        applied_filters: Optional[List[str]] = None,
        metadata_path: Optional[str] = None,
    ):
        self.arrow_folder_root = arrow_folder_root
        self.split = split
        self.processors = processors
        self.modality_registry = modality_registry
        self.transform = transform

        # 1. Arrow data loading (Robust logic inherited)
        arrow_pattern = os.path.join(arrow_folder_root, f"{split}_*")
        arrow_folders = sorted(glob.glob(arrow_pattern))
        
        if not arrow_folders:
            raise ValueError(f"No Arrow data found at {arrow_pattern}")
        
        datasets_list = []
        for p in arrow_folders:
            try:
                datasets_list.append(load_from_disk(p))
            except Exception:
                arrow_files = sorted(glob.glob(os.path.join(p, "*.arrow")))
                if arrow_files:
                    datasets_list.append(load_dataset("arrow", data_files=arrow_files, split="train"))
        
        self.ds = concatenate_datasets(datasets_list)
        self.initial_len = len(self.ds)

        # 2. Smart column selection
        available_cols = self.ds.column_names
        cols_to_keep = ['targetid', 'redshift'] # Keeping redshift for dashboard titles
        for proc in self.processors.values():
            cols_to_keep.extend(proc.required_columns())
        
        # Extract columns mentioned in applied_filters to prevent discarding them during select_columns
        import re
        if applied_filters:
            for expr in applied_filters:
                found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
                for word in found_words:
                    if word in available_cols and word not in cols_to_keep:
                        cols_to_keep.append(word)

        # Remove duplicates and verify existence
        cols_to_keep = list(set([c for c in cols_to_keep if c in available_cols]))
        self.ds = self.ds.select_columns(cols_to_keep)
        self.ds = self.ds.with_format("numpy")

        # 3. Dynamic dataset filtering via applied_filters
        if applied_filters:
            if metadata_path is None:
                from astropt.config import TrainingConfig
                metadata_path = TrainingConfig().metadata_path
            
            # Check if any column is NOT in available_cols (excluding numpy 'np', 'Z', 'redshift' aliases)
            external_cols_needed = False
            for expr in applied_filters:
                found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
                for word in found_words:
                    if word not in available_cols and word not in ['np', 'Z', 'redshift']:
                        external_cols_needed = True
                        break
            
            if external_cols_needed:
                # Dynamic FITS catalog-based ID filtering
                logging.info(f"[{split}] Columns in filters not present in Arrow schema. Performing dynamic FITS catalog matching via '{metadata_path}'...")
                if not metadata_path or not os.path.exists(metadata_path):
                    raise FileNotFoundError(f"Metadata file '{metadata_path}' not found. Cannot evaluate filters: {applied_filters}")
                
                from astropy.table import Table
                logging.info(f"[{split}] Reading FITS catalog...")
                t = Table.read(metadata_path)
                
                id_col = None
                for candidate in ['targetid', 'TARGETID', 'id', 'ID', 'objid', 'OBJID']:
                    if candidate in t.colnames:
                        id_col = candidate
                        break
                if id_col is None:
                    raise KeyError(f"Could not find an ID column in the FITS metadata catalog columns: {t.colnames}")
                
                # Evaluate filters on FITS table
                mask = np.ones(len(t), dtype=bool)
                for expr in applied_filters:
                    py_expr = expr.replace('&&', '&').replace('||', '|')
                    found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
                    eval_namespace = {}
                    for word in found_words:
                        if word in t.colnames:
                            eval_namespace[word] = np.array(t[word])
                    
                    if 'redshift' in t.colnames and 'Z' not in eval_namespace:
                        eval_namespace['Z'] = np.array(t['redshift'])
                    elif 'Z' in t.colnames and 'redshift' not in eval_namespace:
                        eval_namespace['redshift'] = np.array(t['Z'])
                    
                    try:
                        expr_mask = eval(py_expr, {"np": np}, eval_namespace)
                        if isinstance(expr_mask, np.ndarray):
                            mask &= expr_mask
                        else:
                            mask &= np.array(expr_mask, dtype=bool)
                    except Exception as e:
                        raise ValueError(f"Error evaluating filter '{expr}' on FITS catalog: {e}")
                
                allowed_ids = np.array(t[id_col][mask])
                logging.info(f"[{split}] FITS filter matched {len(allowed_ids)} / {len(t)} records.")
                
                # Convert to a set for speed or use np.isin
                allowed_ids_set = set(allowed_ids.astype(int))
                
                def fits_id_filter(targetid_batch):
                    return np.isin(targetid_batch, allowed_ids)
                
                initial_len = len(self.ds)
                self.ds = self.ds.filter(fits_id_filter, batched=True, input_columns=['targetid'])
                final_len = len(self.ds)
                logging.info(f"[{split}] ID-based Arrow filter retained {final_len} / {initial_len} samples ({final_len/initial_len:.1%}).")
            else:
                # High-speed local Arrow filtering
                for expr in applied_filters:
                    py_expr = expr.replace('&&', '&').replace('||', '|')
                    logging.info(f"[{split}] Applying dynamic dataset filter: {expr} (parsed as: {py_expr})")
                    
                    found_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
                    filter_cols = []
                    for word in found_words:
                        if word in available_cols and word not in filter_cols:
                            filter_cols.append(word)
                    
                    if 'Z' in found_words and 'redshift' in available_cols and 'redshift' not in filter_cols:
                        filter_cols.append('redshift')
                    if 'redshift' in found_words and 'Z' in available_cols and 'Z' not in filter_cols:
                        filter_cols.append('Z')
                    
                    def filter_fn(*args):
                        eval_namespace = {}
                        for col, val in zip(filter_cols, args):
                            eval_namespace[col] = np.array(val)
                        
                        if 'redshift' in eval_namespace and 'Z' not in eval_namespace:
                            eval_namespace['Z'] = eval_namespace['redshift']
                        elif 'Z' in eval_namespace and 'redshift' not in eval_namespace:
                            eval_namespace['redshift'] = eval_namespace['Z']
                        
                        try:
                            mask = eval(py_expr, {"np": np}, eval_namespace)
                            if isinstance(mask, np.ndarray):
                                return mask
                            else:
                                return np.array(mask, dtype=bool)
                        except Exception as e:
                            logging.error(f"Error evaluating filter expression '{py_expr}': {e}")
                            raise ValueError(f"Invalid filter expression: '{expr}'. Error: {e}")
                    
                    initial_len = len(self.ds)
                    self.ds = self.ds.filter(filter_fn, batched=True, input_columns=filter_cols)
                    final_len = len(self.ds)
                    logging.info(f"[{split}] Filter retained {final_len} / {initial_len} samples ({final_len/initial_len:.1%}).")

        logging.info(f"[{split}] Multimodal Dataset loaded: {len(self.ds)} samples.")

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _find_matching_indices(source_list: List[int], target_list: List[int]) -> List[int]:
        target_dict = {val: idx for idx, val in enumerate(target_list)}
        return [target_dict[val] for val in source_list if val in target_dict]

    @staticmethod
    def normalise_by_const(x: torch.Tensor, const: float) -> torch.Tensor:
        """
        Normalizes input by dividing by a fixed constant (e.g., global P99).
        """
        x_32 = x.float()
        div_val = const if const > 1e-8 else 1.0
        x_norm = x_32 / div_val
        return x_norm.to(x.dtype)
    
    @staticmethod
    def normalise_zscore(x: torch.Tensor) -> torch.Tensor:
        """
        Standardizes the input tensor (Mean 0, Std 1) while preserving the original dtype.
        """
        x_32 = x.float()
        std, mean = torch.std_mean(x_32, dim=1, keepdim=True)
        x_norm = (x_32 - mean) / (std + 1e-8)
        return x_norm.to(x.dtype)

    @staticmethod
    def normalise_asinh(x: torch.Tensor, a: float = 1.0, c: float = 1.0) -> torch.Tensor:
        """
        Applies Inverse Hyperbolic Sine (asinh) transformation.
        Linear near 0, Logarithmic for high values. Handles negatives gracefully.
        """
        return (torch.asinh(x.float() / a) / c).to(x.dtype)

    @staticmethod
    def data_transforms(
        stage: str = "val",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates the dictionary of data transformations dynamically based on the configuration.
        Expected kwargs: norm_type_{mod_name}, norm_scaler_{mod_name}, norm_const_{mod_name}
        """
        def get_norm_transform(n_type: str, n_scaler: float, n_const: float):
            if n_type == "constant":
                return transforms.Lambda(lambda x: MultimodalDatasetArrow.normalise_by_const(x, n_const))
            elif n_type == "z_score":
                return transforms.Lambda(MultimodalDatasetArrow.normalise_zscore)
            elif n_type == "asinh":
                return transforms.Lambda(lambda x: MultimodalDatasetArrow.normalise_asinh(x, a=n_scaler, c=n_const))
            else:
                return transforms.Lambda(lambda x: x)

        transform_dict = {}
        
        # Extract all unique modality names from kwargs (keys starting with norm_type_)
        mod_names = [k[10:] for k in kwargs.keys() if k.startswith("norm_type_")]
        
        for mod in mod_names:
            n_type = kwargs.get(f"norm_type_{mod}", "z_score")
            n_scaler = kwargs.get(f"norm_scaler_{mod}", 1.0)
            n_const = kwargs.get(f"norm_const_{mod}", 1.0)
            
            # Use specific key names expected by processors
            # Legacy: EuclidImageProcessor expects "images_norm", DESISpectrumProcessor expects "spectra"
            # General: They can also just look for their own name
            norm_tf = get_norm_transform(n_type, n_scaler, n_const)
            
            # Map to legacy keys for compatibility with existing processors
            if mod == "images" or mod == "EuclidImage":
                transform_dict["images_norm"] = norm_tf
            if mod == "spectra" or mod == "DESISpectrum":
                transform_dict["spectra"] = transforms.Compose([norm_tf])
            
            # Always add the specific name as well
            transform_dict[mod] = norm_tf

        return transform_dict

    @staticmethod
    def process_modes(
        batch_data: Dict[str, Any], 
        modality_registry: Any, 
        device: torch.device, 
        shuf: bool = False,
        use_token_mixing: bool = False,
        token_mixing_seed: Optional[int] = None,
        use_cls_token: bool = True,
        cls_position: str = "last"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prepares the batch for training: moves to GPU and handles sequence shifting (X, Y) for autoregressive training.
        """
        modes = modality_registry.generate_sequence(shuf=shuf)

        data_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch_data.items()
        }

        X = {}
        Y = {}

        num_modes = len(modes)

        for i, mode in enumerate(modes):
            data = data_on_device[mode]
            pos = data_on_device[f"{mode}_positions"]
            
            mod_config = modality_registry.get_config(mode)
            
            if getattr(mod_config, 'embed_pos', False):
                b, s = pos.shape[:2]
                pos = torch.arange(s, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)

            if use_token_mixing:
                X[mode] = data
                X[f"{mode}_positions"] = pos
                Y[mode] = data
            else:
                if i == 0 and not use_cls_token:
                    # If it is the first modality and there is no CLS, the target is the shifted sequence
                    Y[mode] = data[:, 1:]
                else:
                    Y[mode] = data
    
                if not use_token_mixing and not use_cls_token:
                    if i == 0 and num_modes > 1:
                        X[mode] = data 
                        X[f"{mode}_positions"] = pos
                    else:
                        X[mode] = data[:, :-1] 
                        X[f"{mode}_positions"] = pos[:, :-1]
                else:
                    X[mode] = data
                    X[f"{mode}_positions"] = pos

            if f"{mode}_aperture" in data_on_device:
                aperture = data_on_device[f"{mode}_aperture"]
                if not use_token_mixing and not use_cls_token:
                    if i == 0 and num_modes > 1:
                        X[f"{mode}_aperture"] = aperture
                    else:
                        X[f"{mode}_aperture"] = aperture[:, :-1]
                else:
                    X[f"{mode}_aperture"] = aperture

        if use_token_mixing and token_mixing_seed is not None:
            X["token_mixing_seed"] = token_mixing_seed

        return {"X": X, "Y": Y}

    @staticmethod
    def prepare_batch(
        batch_data: Dict[str, Any], 
        modality_registry: Any, 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares the batch for INFERENCE (full sequence, no shifting).
        """
        X = {}
        data_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch_data.items()
        }

        for key, val in data_on_device.items():
            # Fix positions if they are embedded (from Float to Long)
            if key.endswith("_positions"):
                mode = key.replace("_positions", "")
                try:
                    mod_config = modality_registry.get_config(mode)
                    if getattr(mod_config, 'embed_pos', False):
                        b, s = val.shape[:2]
                        val = torch.arange(s, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
                except:
                    pass
            X[key] = val

        return X

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample from the arrow dataset by index.
        Uses a retry loop instead of recursion to prevent stack overflow.
        """
        max_retries = 10
        current_idx = idx
        
        for attempt in range(max_retries):
            if torch.is_tensor(current_idx):
                current_idx = current_idx.tolist()

            try:
                item = self.ds[current_idx]
                targetid = item.get('targetid')
                if targetid is not None:
                    targetid = int(targetid)
                
                sample = {
                    "idx": current_idx,
                    "targetid": targetid,
                    "redshift": item.get('redshift')
                }

                # ORCHESTRATION: Iterate over registered processors
                for name, processor in self.processors.items():
                    try:
                        if processor.is_available(item):
                            # The processor knows which configuration to extract from the registry
                            config = self.modality_registry.get_config(name)
                            modality_obj = processor.process(item, self.transform, config)
                            
                            if modality_obj:
                                # Save processed data (flux and positions)
                                sample[name] = modality_obj.flux
                                sample[f"{name}_positions"] = modality_obj.positions
                                if hasattr(modality_obj, "aperture_indices"):
                                    sample[f"{name}_aperture"] = modality_obj.aperture_indices
                    except Exception as e:
                        logging.error(f"Error processing modality {name} for idx {current_idx}: {e}")

                # Safety check: if there is no data for any modality, resample
                has_data = any(name in sample for name in self.processors.keys())
                if not has_data:
                    logging.debug(f"Idx {current_idx} (targetid {targetid}) has no data for requested modalities. Re-sampling...")
                    current_idx = np.random.randint(0, len(self))
                    continue

                return sample

            except (Exception, RecursionError) as e:
                logging.error(f"Error loading idx {current_idx}: {e}")
                current_idx = np.random.randint(0, len(self))
                continue
                
        raise RuntimeError(f"Failed to load a valid sample after {max_retries} retries.")
