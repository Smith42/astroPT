"""
Full definition of a GPT s/Language/Observation Model adapted for float inputs
and with a regression loss, all within this single file.
References:
0) the original nanoGPT code from Andrej Karpathy:
https://github.com/karpathy/nanoGPT
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) Aspia Space's earthPT code:
https://github.com/aspiaspace/earthpt
"""

import math
import inspect
import sys
import random
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks
    flex_attention_avail = True
except:
    print("WARNING: only causal attention is available. Flex Attention requires PyTorch >= 2.6")
    flex_attention_avail = False
from torch.nn import functional as F
from einops.layers.torch import Rearrange

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def generate_prefix_lm_mask(prefix_length):
    """
    Generates a prefix LM causal attention mask.
    From the attention gym 
    https://github.com/pytorch-labs/attention-gym/blob/bbf437e9ea7d802c0ee71d067787f7b57605f9ff/attn_gym/masks/prefix_lm.py

    Args:
        prefix_length: The length of the prefix.

    Note:
        This mask allows full attention within the prefix (first PREFIX_LENGTH tokens)
        and causal attention for the rest of the sequence.
    """

    def prefix_mask(b, h, q_idx, kv_idx):
        return kv_idx < prefix_length

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    prefix_lm_causal_mask = or_masks(prefix_mask, causal_mask)
    prefix_lm_causal_mask.__name__ = f"prefix_lm_causal_mask_{prefix_length}"
    return prefix_lm_causal_mask

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.attn_type = config.attn_type
        if self.attn_type == "causal":
            # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))
        elif (self.attn_type == "prefix" and flex_attention_avail):
            # flex attention also make GPU brrrr for non-causal masking, only available with DDP for PyTorch >= 2.6
            # need to compile flex attention for performance!
            self.flex_attention = torch.compile(flex_attention)
        else:
            raise NotImplementedError("Attention type must be one of 'causal' or 'prefix'. Prefix requires PyTorch >= 2.6.")

    def forward(self, x, block_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.attn_type == "causal":
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        elif self.attn_type == "prefix":
            y = self.flex_attention(q, k, v, block_mask=block_mask)
        else: 
            raise NotImplementedError("Attention type must be one of 'causal' or 'prefix'. Prefix requires PyTorch >= 2.6.")
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, block_mask=None):
        x = x + self.attn(self.ln_1(x), block_mask=None)
        x = x + self.mlp(self.ln_2(x))
        return x

class KSparseLayer(nn.Module):
    """Implements k-sparsity with overcomplete representation using embedding_bag for efficient decoding
    with mechanisms to prevent dying neurons through forward integration of dead neurons"""

    def __init__(self, config):
        super().__init__()
        self.k = int(config.n_embd * config.k_ratio)  # number of activations to keep
        self.kaux = int(config.n_embd * 0.2)  # number of dead latents to use (20% of embedding dimension)
        self.n_embd = config.n_embd
        self.overcomplete_size = 4 * config.n_embd
        
        # Linear projections with overcomplete middle layer
        self.to_overcomplete = nn.Linear(config.n_embd, self.overcomplete_size, bias=config.bias)
        self.from_overcomplete = nn.Linear(self.overcomplete_size, config.n_embd, bias=config.bias)
        # Layer norm for numerical stability
        self.norm = LayerNorm(self.overcomplete_size, bias=config.bias)

        # For tracking activations
        self.buffer_size = 1024
        self.register_buffer('activation_counts', torch.zeros(self.overcomplete_size))
        self.register_buffer('activation_last_used', torch.zeros(self.overcomplete_size))
        self.register_buffer('recent_inputs', torch.zeros(self.buffer_size, config.n_chan, config.image_size, config.image_size))
        self.register_buffer('activation_indices', torch.zeros(self.buffer_size, dtype=torch.long) - 1)
        self.register_buffer('activation_values', torch.zeros(self.buffer_size))
        self.input_ptr = 0
        self.iteration = 0
        self.dead_neuron_factor = 1.0/32.0  # Alpha in 2406.04093v1 A.2
        self.dead_threshold = 5000  # Consider a neuron dead if not used for this many iterations
        self.noise_factor = 0.01  # Small noise to break ties and wake up neurons

    def get_top_activations(self, k=10):
        """Returns the indices of the k most common activations"""
        return torch.topk(self.activation_counts, k)
    
    def get_example_inputs(self, feature_idx):
        """Returns stored inputs that strongly activated a given feature"""
        matches = (self.activation_indices == feature_idx)
        return self.recent_inputs[matches], self.activation_values[matches]
    
    def get_dead_neuron_indices(self):
        """Identify neurons that haven't been activated for a while"""
        if self.iteration < self.dead_threshold:
            return torch.tensor([], device=self.activation_last_used.device, dtype=torch.long)
        
        # Neurons that haven't been used for dead_threshold iterations
        dead_mask = (self.iteration - self.activation_last_used) > self.dead_threshold
        dead_indices = torch.where(dead_mask)[0]
        return dead_indices

    def forward(self, x, img=None):
        self.iteration += 1
        batch_size, seq_len, _ = x.shape
        
        # Project to overcomplete space and normalize
        h = self.norm(self.to_overcomplete(x))
        
        # Add small noise to break ties and help wake up neurons during training
        if self.training:
            h = h + torch.randn_like(h) * self.noise_factor
        
        # Get top k activations in overcomplete space
        values, indices = torch.topk(h, self.k, dim=-1)

        summed_values = []
        for B in range(len(h)):
            summed_values.append(
                torch.bincount(indices[B].flatten(), weights=values[B].flatten(), minlength=h.shape[-1])
            )
        summed_values = torch.stack(summed_values)
        topk_summed_values, topk_summed_indices = torch.topk(summed_values, self.k, dim=-1)
        
        # revive dead neurons in a way analagous to the k aux loss in 2406.04093
        dead_contribution = None
        if self.training:
            dead_indices = self.get_dead_neuron_indices()
            # use dead neurons if we have any
            if len(dead_indices) > 0:
                kaux = min(self.kaux, len(dead_indices))
                if kaux > 0:
                    # get the activations for dead neurons
                    h_dead = h[:, :, dead_indices[:kaux]]
                    # extract weights for dead neurons 
                    dead_weights = self.from_overcomplete.weight.t()[dead_indices[:kaux]]
                    # calculate the contribution from dead neurons (but don't perform topk on them)
                    dead_contribution = torch.matmul(h_dead, dead_weights) * self.dead_neuron_factor
        
        # Update activation statistics
        if self.training:
            # mark when neurons were last used
            self.activation_last_used[indices.reshape(-1)] = self.iteration
            # count which features are being activated
            unique_indices, counts = torch.unique(indices, return_counts=True)
            batch_counts = torch.zeros_like(self.activation_counts)
            batch_counts[unique_indices] = counts.float()
            self.activation_counts.mul_(0.99).add_(batch_counts * (1 - 0.99))
        if img is not None:
            # Store recent inputs that strongly activated each feature
            for b in range(topk_summed_indices.size(0)):
                strongest_activation = topk_summed_indices[b, 0]  # Take first occurrence
                strongest_value = topk_summed_values[b, 0]
                # Store the input that caused this activation
                self.recent_inputs[self.input_ptr] = img[b].detach()
                self.activation_indices[self.input_ptr] = strongest_activation
                self.activation_values[self.input_ptr] = strongest_value
                # Move pointer forward, wrapping around when we hit buffer size
                self.input_ptr = (self.input_ptr + 1) % self.buffer_size

        # Reshape indices and values for embedding_bag
        batch_size, seq_len, _ = indices.shape
        indices = indices.reshape(batch_size * seq_len, self.k)
        values = values.reshape(batch_size * seq_len, self.k)
        
        # Use embedding_bag for efficient sparse decoding
        # (see https://x.com/norabelrose/status/1887585218145755581)
        decoded = F.embedding_bag(
            indices, 
            self.from_overcomplete.weight.t(),  # Transpose weight matrix to match expected shape
            per_sample_weights=values,
            mode="sum"
        ).reshape(batch_size, seq_len, -1)
        
        # add contribution from dead neurons if available
        if dead_contribution is not None and self.training:
            decoded = decoded + dead_contribution

        return decoded

@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_chan: int = 1
    dropout: float = 0.0
    patch_size: int = 16
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    k_ratio: float = 0.05 # Number of sparse ks to keep. If k_ratio == 0 disable sparsity.
    image_size: int = 512 # Size of an image used during training.
    attn_type: str = "causal" # causal or prefix

class GPT(nn.Module):

    def __init__(self, config, master_process=True):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        # Split transformer blocks into pre and post sparse sections
        n_pre = config.n_layer // 2
        n_post = config.n_layer - n_pre

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Sequential(
                 nn.Linear(config.patch_size*config.patch_size*config.n_chan, config.n_embd),
                 nn.ReLU(),
                 nn.Linear(config.n_embd, config.n_embd),
                 nn.ReLU(),
            ),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            pre_blocks = nn.ModuleList([Block(config) for _ in range(n_pre)]),
            post_blocks = nn.ModuleList([Block(config) for _ in range(n_post)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Add central sparse layer
        self.sparse = KSparseLayer(config) if config.k_ratio > 0 else nn.Identity()
        
        self.lm_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd*2),
            nn.ReLU(),
            nn.Linear(config.n_embd*2, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.patch_size*config.patch_size*config.n_chan),
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # TODO rethink weight tying
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        self.master_process = master_process
        if self.master_process: print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, prefix_len=None, raw_im=None):
        device = idx.device
        b, t, c = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if self.config.attn_type == "prefix":
            # TODO we need to make sure that the prefix hyperparameters are tuned well:
            if prefix_len is None:
                # if we don't pass a prefix length assume we want it sampled at random
                # TODO do we want this to be an eval mode switch?
                prefix_len = random.randrange(self.config.block_size - 1)
            prefix_lm_mask = generate_prefix_lm_mask(prefix_len)
            block_mask = create_block_mask(prefix_lm_mask, None, None, self.config.block_size, self.config.block_size)
        else:
            block_mask = None
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.pre_blocks:
            x = block(x, block_mask=block_mask)
        x = self.sparse(x, img=raw_im)  # Central sparse layer
        for block in self.transformer.post_blocks:
            x = block(x, block_mask=block_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # if we have prefix attention on we only want to backprop through
            # tokens where our model cannot look ahead! so we mask the loss:
            prefix_mask = torch.ones_like(targets, dtype=torch.bool)
            if self.config.attn_type == "prefix":
                prefix_mask[:, :prefix_len] = False
            loss = F.huber_loss(logits[prefix_mask], targets[prefix_mask])
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def get_embeddings(self, idx, prefix_len=None):
        device = idx.device
        b, t, ch = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if self.config.attn_type == "prefix":
            # TODO we need to make sure that the prefix hyperparameters are tuned well:
            if prefix_len is None:
                # if we don't pass a prefix length assume we want it sampled at random
                # TODO do we want this to be an eval mode switch?
                prefix_len = random.randrange(self.config.block_size - 1)
            prefix_lm_mask = generate_prefix_lm_mask(prefix_len)
            block_mask = create_block_mask(prefix_lm_mask, None, None, self.config.block_size, self.config.block_size)
        else:
            block_mask = None
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, block_mask=block_mask)
        embeddings = x

        return embeddings

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        if self.master_process: print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, new_tokens, temperature=0.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for i in range(new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            idx_next = logits + (torch.randn(logits.size())*temperature).to(logits.device)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_embeddings(self, idx, average_type="mean", prefix_len=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and get the embedding from the transformer model for that series.

        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.
        """
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        embeddings = self.get_embeddings(idx_cond, prefix_len=None)
        if average_type == "mean":
            # We only care about the average embedding
            return torch.mean(embeddings, dim=1)
        elif average_type == "exp_decay":
            weights = torch.logspace(0, -1, embeddings.shape[1], device=embeddings.device).unsqueeze(0).unsqueeze(-1)
            return torch.sum(weights*embeddings, dim=1)/torch.sum(embeddings, dim=1)
        elif average_type == "none":
            return embeddings
        else:
            raise NotImplementedError
