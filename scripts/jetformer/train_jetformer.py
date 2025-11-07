# train_jetformer.py
import math
import os
import csv
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

# NEW: Configure matplotlib to use a non-interactive backend
# This must be done BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================================================================================
# Block 1: Configuration
# ======================================================================================
@dataclass
class CFG:
    # --- Heavier Transformer Config ---
    d_model: int = 1536
    n_heads: int = 24
    n_layers: int = 16
    
    # --- Training Config ---
    epochs: int = 2000
    batch_size: int = 64
    lr: float = 3e-4 # Constant learning rate
    wd: float = 0.01
    
    # --- Dataset Switch ---
    # Set this to "car" or "galaxy" to choose the dataset
    dataset_name: str = "galaxy" 
    
    # --- Other Model/Data Params ---
    img_size: int = 32 # This MUST stay 32x32 for the JetFormerLite model
    in_ch: int = 3
    patch: int = 4
    n_tokens: int = (img_size // patch)**2
    d_token: int = in_ch * patch * patch
    gmm_K: int = 4
    flow_steps: int = 4
    
    # --- Stability & Checkpointing ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip_val: float = 1.0
    
    # These paths are now set dynamically in the train() function
    checkpoint_path: str = "" 
    samples_dir: str = ""
    loss_csv_path: str = ""
    loss_plot_path: str = ""
    
    # --- Noise Curriculum ---
    noise_max: float = 0.1
    noise_min: float = 0.0

# ======================================================================================
# Block 2: Loss Logging and Plotting Utilities
# ======================================================================================

def append_losses_to_csv(epoch, train_loss, val_loss, filename):
    """Appends the epoch, average train loss, and optional val loss to a CSV file."""
    # Check if the file exists to write headers
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss']) 
        
        # Write the data. csv.writer handles None as an empty field.
        writer.writerow([epoch, train_loss, val_loss]) 

def plot_loss_from_csv(csv_path, output_path):
    """Reads a CSV file and saves a plot of the train and validation loss curves."""
    # Prevent error if the file doesn't exist yet
    if not os.path.isfile(csv_path):
        return
    
    # Read the data using pandas. 
    # Empty val_loss fields will be read as NaN.
    df = pd.read_csv(csv_path) 
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training loss (all epochs)
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    

    # 1. Create a new DataFrame containing only rows where 'val_loss' is NOT NaN
    df_val = df.dropna(subset=['val_loss'])
    
    # 2. Plot the filtered data. 
    # This will plot just the valid points (e.g., 5, 10, 15...)
    # and connect them with a dashed line.
    if not df_val.empty: # Only plot if we have at least one validation point
        ax.plot(df_val['epoch'], df_val['val_loss'], 
                label='Validation Loss', color='orange', 
                linestyle='--', marker='o') 
    
    ax.set_title('Training and Validation Loss per Epoch') 
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.legend() 
    ax.grid(True)
    
    # Save the plot and close the figure to free memory
    fig.savefig(output_path)
    plt.close(fig)

# ======================================================================================
# Block 3: Data Loading
# ======================================================================================

def get_car_dataloader(cfg):
    """Loads the CIFAR-10 dataset and filters it to only include the 'car' class."""
    print("Loading and filtering CIFAR-10 for 'car' class...")
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    
    # Load the full training dataset
    full_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # The 'car' class is at index 1 in CIFAR-10
    car_class_index = 1
    
    # Find the indices of all images that belong to the car class
    car_indices = [i for i, (_, label) in enumerate(full_dataset) if label == car_class_index]
    
    # Create a Subset of the original dataset using only the car indices
    car_subset = Subset(full_dataset, car_indices)
    
    print(f"Found {len(car_subset)} images of cars.")
    
    # Create a DataLoader from the subset
    return DataLoader(car_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

def get_galaxy_dataloader(cfg, folder="galaxy_pt"):
    """
    Loads preprocessed 32x32 CHW uint8 tensors from .pt shards and feeds the GPU fast.
    Output: dict {"img": FloatTensor[B,3,32,32] in [0,1], "label": LongTensor[B]}
    """
    import os, glob, torch
    from torch.utils.data import IterableDataset, DataLoader, get_worker_info

    files = sorted(glob.glob(os.path.join(folder, "shard_*.pt")))
    if not files:
        raise FileNotFoundError(f"No shards found in {folder}. Run make_galaxy_pt_shards.py first.")

    class PTShardStream(IterableDataset):
        def __init__(self, files):
            self.files = files

        def __iter__(self):
            info = get_worker_info()
            # shard files across workers
            if info is None:
                my_files = self.files
            else:
                my_files = self.files[info.id::info.num_workers]

            for f in my_files:
                data = torch.load(f, map_location="cpu")  # {"images": uint8 [N,3,32,32]}
                imgs_u8 = data["images"]                 # NCHW uint8
                # yield per-sample; DataLoader will stack
                for i in range(imgs_u8.size(0)):
                    # convert to float [0,1] here; no PIL, no resize, no GPU stall
                    yield {"img": imgs_u8[i].float().div_(255.0), "label": 0}

    def _collate(batch):
        imgs = torch.stack([b["img"] for b in batch], dim=0)  # [B,3,32,32] float
        labels = torch.zeros(len(batch), dtype=torch.long)
        return {"img": imgs, "label": labels}

    # multiple workers safe (we shard files, not streams)
    nw = min(8, max(2, (os.cpu_count() or 4) - 1))
    return DataLoader(
        PTShardStream(files),
        batch_size=cfg.batch_size,
        num_workers=nw,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=_collate,
    )
# ======================================================================================
# Block 4: Checkpointing Functions
# ======================================================================================
def save_checkpoint(epoch, model, optimizer, cfg):
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, cfg.checkpoint_path)

def load_checkpoint(model, optimizer, cfg):
    if not os.path.exists(cfg.checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return 0
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return start_epoch

# ======================================================================================
# Block 5.1: Image and Token Utilities
# This block contains helper functions for preprocessing images and converting them
# into a sequence of "tokens" that the Transformer can understand, and back again.
# ======================================================================================
def uniform_dequantize(x: torch.Tensor) -> torch.Tensor:
    """
    Takes a tensor of pixel values (scaled 0-1) and makes them continuous.
    It adds a tiny amount of uniform noise, breaking the discrete nature of pixel values.
    This is a crucial step for training continuous models like normalizing flows.
    """
    return (x + torch.rand_like(x) / 256.0).clamp(0.0, 1.0)

def patchify(x: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Converts a batch of images into a sequence of flattened patches (tokens).
    It slices the image into a grid and then flattens each patch.
    Input Shape: (Batch, Channels, Height, Width) -> (B, C, H, W)
    Output Shape: (Batch, Num_Patches, Patch_Dimension) -> (B, N, D_token)
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by the patch size."
    
    # Use 'unfold' to create sliding blocks (patches) across height and width
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Reshape and flatten to get the final sequence of tokens
    x = x.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_size * patch_size)
    return x

def depatchify(tokens: torch.Tensor, C: int = 3, H: int = 32, W: int = 32, patch_size: int = 4) -> torch.Tensor:
    """
    The exact inverse of the 'patchify' function.
    Converts a sequence of tokens back into an image format.
    """
    B, N, D = tokens.shape
    hp, wp = H // patch_size, W // patch_size # Number of patches along height and width
    
    # Reshape the sequence back into a grid of patches
    x = tokens.reshape(B, hp, wp, C, patch_size, patch_size)
    
    # Permute and reshape to reconstruct the final image
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return x
    
# ======================================================================================
# Block 5.2: Normalizing Flow (TinyFlow)
# This block defines a simple RealNVP-style normalizing flow. A flow is an invertible
# neural network, meaning it can map data to a latent space and back again perfectly.
# Crucially, it provides the log-determinant of the Jacobian ('logdet'), which is
# needed for the change of variables formula to calculate the exact likelihood of data.
# Its purpose here is to "pre-process" the complex image distribution into a simpler one
# that is easier for the Transformer to model.
# ======================================================================================
class CouplingNet(nn.Module):
    """A small convolutional network that predicts the scale (s) and shift (t) parameters for the Affine Coupling Layer."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, channels * 2, kernel_size=3, padding=1) # Output has 2x channels for s and t
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        st = self.net(x)
        C = x.size(1)
        s, t = st[:, :C], st[:, C:] # Split the output into scale and shift
        s = torch.tanh(s) * 1.5 # Bound the scale for numerical stability
        return s, t

class AffineCoupling(nn.Module):
    """
    An affine coupling layer. It splits the input using a mask. One part is left
    unchanged (identity), and this part is used to predict the scale/shift that
    will transform the *other* part of the input. This makes the transformation
    powerful yet easily invertible.
    """
    def __init__(self, in_ch: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask) # A binary mask (e.g., checkerboard)
        self.net = CouplingNet(in_ch)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_id = x * self.mask  # The part that remains unchanged
        s, t = self.net(x_id) # Predict s and t from the unchanged part

        if not reverse: # Forward pass: x -> z
            # Transform the other part of x: y = x * scale + shift
            y = x_id + (1 - self.mask) * (x * torch.exp(s) + t)
            # The log-determinant is just the sum of the logs of the scale factors
            logdet = ((1 - self.mask) * s).flatten(1).sum(dim=1)
            return y, logdet
        else: # Inverse pass: z -> x
            # The inverse is cheap to compute: x = (y - shift) / scale
            y = x_id + (1 - self.mask) * ((x - t) * torch.exp(-s))
            logdet = -((1 - self.mask) * s).flatten(1).sum(dim=1)
            return y, logdet

def checker_mask(C: int, H: int, W: int, flip: bool = False, device: str = "cpu") -> torch.Tensor:
    """Creates a checkerboard mask where half the channels are masked."""
    m = torch.zeros(1, C, H, W, device=device)
    m[:, ::2, :, :] = 1.0 # Mask even-indexed channels
    return 1.0 - m if flip else m

class TinyFlow(nn.Module):
    """A stack of Affine Coupling layers. By alternating the mask between layers, we ensure all dimensions get transformed."""
    def __init__(self, in_ch: int, img_size: int, steps: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([AffineCoupling(in_ch, checker_mask(in_ch, img_size, img_size, flip=(k % 2 == 1))) for k in range(steps)])
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.size(0))
        z = x
        
        # Apply the sequence of transformations
        if not reverse:
            for b in self.blocks:
                z, ld = b(z, reverse=False)
                logdet += ld
        else: # Apply in reverse for the inverse pass
            for b in reversed(self.blocks):
                z, ld = b(z, reverse=True)
                logdet += ld
        return z, logdet

# ======================================================================================
# Block 5.3: Autoregressive Transformer (TinyGPT)
# This block defines a standard "decoder-only" Transformer, similar to GPT. Its job is
# to model the sequence of latent tokens produced by the Normalizing Flow.
# It is "autoregressive" because it predicts each token based on all the tokens that came before it.
# ======================================================================================
class CausalSelfAttention(nn.Module):
    """
    The core mechanism of the Transformer. It allows each token to look at all
    previous tokens in the sequence to gather context. A "causal" mask is applied
    to prevent it from "cheating" by looking at future tokens.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.register_buffer("mask", None, persistent=False)
        
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generates or retrieves the triangular mask to enforce causality."""
        if self.mask is None or self.mask.size(0) != T:
            self.mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return self.mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        m = self._get_causal_mask(T, x.device)
        out, _ = self.attn(x, x, x, attn_mask=m, need_weights=False)
        return out

class DecoderBlock(nn.Module):
    """A single Transformer block, which combines causal self-attention and a feed-forward network (MLP)."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x)) # Attention with residual connection
        x = x + self.mlp(self.ln2(x))  # MLP with residual connection
        return x

class TinyGPT(nn.Module):
    """A stack of decoder blocks to form the full Transformer model."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)

# ======================================================================================
# Block 5.4: GMM Output Head and Loss
# Since the latent tokens are continuous, we can't use a standard classification head.
# Instead, this block predicts the parameters of a Gaussian Mixture Model (GMM) for each token.
# A GMM provides a flexible probability distribution over continuous space. The loss
# is then the negative log-likelihood (NLL) of the true token under this predicted distribution.
# ======================================================================================
class GMMHead(nn.Module):
    """
    A linear layer that takes the Transformer's output and maps it to the
    parameters of a GMM for each token in the sequence.
    For each of K components, we predict:
    1. A mixture weight (pi)
    2. A mean vector (mu)
    3. A log standard deviation vector (log_sigma)
    """
    def __init__(self, d_model: int, d_token: int, K: int):
        super().__init__()
        self.K, self.D = K, d_token
        self.proj = nn.Linear(d_model, K * (1 + 2 * d_token))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = h.shape
        out = self.proj(h).view(B, N, self.K, 1 + 2 * self.D)
        
        logits_pi = out[..., 0]          # Mixture weights (in logit form)
        mu = out[..., 1:1+self.D]        # Means
        log_sigma = out[..., 1+self.D:]  # Log standard deviations
        
        log_sigma = torch.clamp(log_sigma, -7, 2) # Clamp for stability
        return logits_pi, mu, log_sigma

def gmm_nll(y: torch.Tensor, logits_pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log-likelihood of target tokens `y` under the predicted GMM.
    This involves calculating the probability of `y` under each Gaussian component,
    weighting it by the mixture probabilities, and using the log-sum-exp trick
    to compute the total log-likelihood in a numerically stable way.
    """
    B, N, D = y.shape; K = logits_pi.size(-1)
    
    y = y.unsqueeze(2) # Reshape for broadcasting with GMM parameters

    # Calculate log probability of y for each Gaussian component: log N(y | mu, sigma)
    inv_var = torch.exp(-2 * log_sigma)
    logp = -0.5 * ((y - mu)**2 * inv_var).sum(-1) - log_sigma.sum(-1) - 0.5 * D * math.log(2 * math.pi)
    
    # Combine with mixture weights and compute total log-likelihood
    logmix = F.log_softmax(logits_pi, dim=-1) + logp
    
    # log-sum-exp over the K components gives the log-likelihood for each token.
    # Sum over the sequence length to get per-sample NLL.
    return -torch.logsumexp(logmix, dim=-1).sum(dim=1)

# ======================================================================================
# Block 5.5: The Complete JetFormerLite Model
# This final class assembles all the previous components into the complete model.
# It defines the full forward pass:
# Image -> Flow -> Latent -> Patchify -> Transformer -> GMM Head -> Loss
# ======================================================================================
class JetFormerLite(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        # 1. The invertible pre-processor
        self.flow = TinyFlow(cfg.in_ch, cfg.img_size, cfg.flow_steps)
        # 2. The tokenizer (projection from raw patch to model dimension)
        self.in_proj = nn.Linear(cfg.d_token, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.n_tokens, cfg.d_model) * 0.01)
        # 3. The autoregressive model
        self.gpt = TinyGPT(cfg.d_model, cfg.n_heads, cfg.n_layers)
        # 4. The output head
        self.head = GMMHead(cfg.d_model, cfg.d_token, cfg.gmm_K)

    def forward(self, x: torch.Tensor, epoch_frac: float = 1.0) -> torch.Tensor:
        # Pre-process image
        if x.dtype == torch.uint8: x = x.float() / 255.0
        x = uniform_dequantize(x)
        
        # 1. Pass image through the flow to get latent `z` and `logdet`
        z, logdet = self.flow(x, reverse=False)
        
        # 2. Convert latent image `z` into a sequence of tokens
        tokens = patchify(z, self.cfg.patch)
        
        # Add annealed noise for training stability
        sigma = self.cfg.noise_max + (self.cfg.noise_min - self.cfg.noise_max) * epoch_frac
        if self.training and sigma > 0:
            tokens = tokens + torch.randn_like(tokens) * sigma
            
        # 3. Project tokens and pass through the Transformer
        h = self.in_proj(tokens) + self.pos
        h = self.gpt(h)
        
        # 4. Predict GMM parameters for the *next* token
        logits_pi, mu, log_sigma = self.head(h[:, :-1])
        target = tokens[:, 1:] # Teacher-forcing
        
        # 5. Calculate the two components of the loss
        nll_gmm = gmm_nll(target, logits_pi, mu, log_sigma) # NLL of latent z
        
        # Final loss: NLL(x) = NLL(z) - log|det J|
        loss = (nll_gmm - logdet).mean()
        return loss

    @torch.no_grad()
    def sample(self, n: int = 16, x_real_batch: torch.Tensor = None): 
        self.eval()
        B = n; N = self.cfg.n_tokens; device = next(self.parameters()).device
        
        # Check if we are doing reconstruction or unconditional generation
        if x_real_batch is None:
            # This is the ORIGINAL unconditional generation path
            tokens = torch.zeros(B, N, self.cfg.d_token, device=device)
            print("Generating tokens autoregressively (unconditional)...")
            
            for t in tqdm(range(N - 1), leave=False):
                h_in = self.in_proj(tokens) + self.pos
                h_out = self.gpt(h_in)
                logits_pi, mu, log_sigma = self.head(h_out[:, t:t+1])
                pi = F.softmax(logits_pi.squeeze(1), dim=-1)
                comp_idx = torch.multinomial(pi, 1)
                gather_idx = comp_idx[:, :, None].expand(-1, 1, self.cfg.d_token)
                sel_mu = mu.squeeze(1).gather(1, gather_idx).squeeze(1)
                sel_sigma = log_sigma.squeeze(1).gather(1, gather_idx).squeeze(1).exp()
                y = sel_mu + torch.randn_like(sel_mu) * sel_sigma
                tokens[:, t+1] = y
                
            z = depatchify(tokens, C=self.cfg.in_ch, H=self.cfg.img_size, W=self.cfg.img_size, patch_size=self.cfg.patch)
            x, _ = self.flow(z, reverse=True)
            x = x.clamp(0, 1)
            return x
        
        else:
            # This is the RECONSTRUCTION path
            print("Reconstructing real images for side-by-side comparison...")
            
            # We'll use n//2 real images and n//2 predicted images to make n total
            n_pairs = n // 2 
            # Get first n/2 images from the provided batch and move to device
            x_real = x_real_batch[:n_pairs].to(device) 

            # Define C, H, and W from the model's configuration
            C = self.cfg.in_ch
            H = self.cfg.img_size
            W = self.cfg.img_size

            
            # 1. Pass real images through the flow to get latent z
            z_real, _ = self.flow(x_real, reverse=False)
            
            # 2. Patchify to get the real tokens
            tokens_real = patchify(z_real, self.cfg.patch)
            
            # 3. Get the model's predictions (teacher-forced)
            # We pass the *entire* real token sequence and get predictions for what *should* come next
            h_in = self.in_proj(tokens_real) + self.pos
            h_out = self.gpt(h_in)
            
            # 4. Get GMM parameters for tokens 1...N-1
            logits_pi, mu, log_sigma = self.head(h_out[:, :-1])
            
            # 5. Create the "predicted" token sequence
            tokens_pred = torch.zeros_like(tokens_real)
            tokens_pred[:, 0] = tokens_real[:, 0] # Copy first token (it's not predicted)
            
            # 6. Get deterministic 'mu' from most likely GMM component
            # This is the model's "best guess" for each token, given the *real* preceding token
            best_comp_idx = torch.argmax(logits_pi, dim=-1, keepdim=True) # Shape: [B, N-1, 1]
            gather_idx = best_comp_idx.unsqueeze(-1).expand(-1, -1, -1, self.cfg.d_token)
            sel_mu = torch.gather(mu, 2, gather_idx).squeeze(2) # Shape: [B, N-1, D_token]
            tokens_pred[:, 1:] = sel_mu # Fill in the predicted tokens
            
            # 7. Depatchify and invert flow for predicted images
            z_pred = depatchify(tokens_pred, C=self.cfg.in_ch, H=self.cfg.img_size, W=self.cfg.img_size, patch_size=self.cfg.patch)
            x_pred, _ = self.flow(z_pred, reverse=True)
            x_pred = x_pred.clamp(0, 1) # This is the final reconstructed image batch
            
            # 8. Interleave real and predicted images for side-by-side saving
            # We stack [x_real, x_pred] and then reshape
            # The saved grid will look like: [x_real[0], x_pred[0], x_real[1], x_pred[1], ...]
            combined = torch.stack([x_real, x_pred], dim=1) # Shape [n_pairs, 2, C, H, W]
            combined = combined.view(n, C, H, W) # Shape [n, C, H, W]
            
            return combined

# ======================================================================================
# Block 6: Main Training Loop
# ======================================================================================
def train():
    # --- Basic Setup ---
    cfg = CFG()
    device = cfg.device
    print(f"Using device: {device}")
    
    # Dynamically set paths based on the dataset name in CFG
    cfg.checkpoint_path = f"checkpoint_{cfg.dataset_name}.pt"
    cfg.samples_dir = f"samples_{cfg.dataset_name}"
    cfg.loss_csv_path = f"loss_log_{cfg.dataset_name}.csv"
    cfg.loss_plot_path = f"loss_plot_{cfg.dataset_name}.png"
    
    os.makedirs(cfg.samples_dir, exist_ok=True)
    print(f"Running experiment: {cfg.dataset_name}")
    print(f"Checkpoints will be saved to: {cfg.checkpoint_path}")

    # --- Model and Optimizer Setup ---
    model = JetFormerLite(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    start_epoch = load_checkpoint(model, opt, cfg)

    # --- Data Loading ---
    # Use the dataset_name from CFG to select the loader
    if cfg.dataset_name == "car":
        loader = get_car_dataloader(cfg)
    elif cfg.dataset_name == "galaxy":
        loader = get_galaxy_dataloader(cfg)
    else:
        raise ValueError(f"Unknown dataset in CFG: '{cfg.dataset_name}'")
    
    # We can't know the steps_per_epoch for a streaming dataset,
    # so we'll just use a large number for the epoch_frac calculation.
    # This is only for the noise curriculum, so it doesn't need to be exact.
    # 5000 (car) / 64 = ~79. 1,000,000 (galaxy) / 64 = ~15625
    steps_per_epoch = 15625 if cfg.dataset_name == "galaxy" else len(loader)
    
    print(f"Starting training from epoch {start_epoch} up to {cfg.epochs}...")
    # --- Main Training Loop ---
    for ep in range(start_epoch, cfg.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{cfg.epochs}")
        
        # Accumulate losses to calculate an average for the epoch
        epoch_losses = []
        
        for i, batch in enumerate(pbar):
            # The car loader returns (img, _), the galaxy loader returns {"img": ..., "label": ...}
            # We standardize this here.
            if isinstance(batch, dict):
                img = batch["img"]
            else:
                img = batch[0]
                
            current_step = ep * steps_per_epoch + i
            
            # Forward and backward pass
            img = img.to(device)
            epoch_frac = current_step / (cfg.epochs * steps_per_epoch)
            loss = model(img, epoch_frac=epoch_frac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_val)
            opt.step()
            
            # Add batch loss to the list
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        
        # --- End-of-Epoch Operations ---
        # Calculate the average loss for the epoch
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"Epoch {ep+1} finished. Average Train Loss: {avg_train_loss:.3f}")

        # Save a checkpoint
        save_checkpoint(ep, model, opt, cfg)
        
        # Initialize avg_val_loss to None
        avg_val_loss = None
        
        # Generate samples and RUN VALIDATION every 5 epochs
        if (ep + 1) % 5 == 0:
            print(f"--- Running Validation & Sampling for epoch {ep+1} ---")
            
            # Run validation on the entire test set
            model.eval() # Set model to evaluation mode
            test_shard_path = os.path.join("galaxy_pt", "shard_0000_test.pt")
            val_losses = []
            test_images_f32 = None # To store loaded test images
            try:
                # 1. Load the full test shard data
                test_data = torch.load(test_shard_path, map_location='cpu')
                test_images_u8 = test_data["images"]
                test_images_f32 = test_images_u8.float().div_(255.0) # [N, C, H, W]
                
                # 2. Create a DataLoader for the test set
                test_dataset = TensorDataset(test_images_f32)
                test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
                
                print(f"Running validation on {len(test_dataset)} test images...")
                val_pbar = tqdm(test_loader, desc="Validation", leave=False)
                
                with torch.no_grad(): # Disable gradient calculation
                    for val_batch in val_pbar:
                        val_img = val_batch[0].to(device)
                        # Run model forward pass, epoch_frac=1.0 means no noise
                        val_loss = model(val_img, epoch_frac=1.0) 
                        val_losses.append(val_loss.item())
                
                # 3. Calculate average validation loss
                if val_losses:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    print(f"Validation finished. Average Validation Loss: {avg_val_loss:.3f}")
                
            except FileNotFoundError:
                print(f"WARNING: Test shard '{test_shard_path}' not found. Skipping validation.")
            except Exception as e:
                print(f"WARNING: Error during validation: {e}. Skipping validation.")
            
            model.train() # Set model back to training mode
            # --- END OF VALIDATION CODE ---

            # Use the test data we already loaded for sampling
            test_batch = None
            if test_images_f32 is not None:
                try:
                    # Get 16 random indices from the loaded test images
                    num_images_in_shard = test_images_f32.size(0)
                    random_indices = torch.randperm(num_images_in_shard)[:16]
                    
                    # Select the 16 random images
                    test_batch = test_images_f32[random_indices]

                except Exception as e:
                    print(f"WARNING: Error selecting random samples: {e}. Skipping sample generation.")
            else:
                 print("WARNING: test_images_f32 not loaded (validation may have failed). Skipping sample generation.")
            # --- END OF MODIFICATION ---

            # Only proceed if we successfully loaded the test_batch
            if test_batch is not None:
                # This part is UNCHANGED as requested
                fake_images = model.sample(n=32, x_real_batch=test_batch)
                sample_path = os.path.join(cfg.samples_dir, f"epoch_{ep+1:03d}.png")
                save_image(fake_images, sample_path, nrow=2)
                print(f"Samples saved to {sample_path}")

        # Log the average losses to the CSV file
        #    avg_val_loss is either None (if not epoch 5) or the calculated value
        append_losses_to_csv(ep + 1, avg_train_loss, avg_val_loss, cfg.loss_csv_path)

        # Update the static plot of the loss curve
        plot_loss_from_csv(cfg.loss_csv_path, cfg.loss_plot_path)

    print("Training finished.")

if __name__ == "__main__":
    train()