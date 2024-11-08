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
from dataclasses import dataclass

import torch
import torch.nn as nn
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

class CausalSelfAttention(nn.Module):

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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

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
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Encoder(nn.Module):
    """ base module to move from data space to embedding space """
     
    def __init__(self, config, in_size):
        super().__init__()
        self.c_fc    = nn.Linear(in_size, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x

class Decoder(nn.Module):
    """ base module to move from embedding space to data space  """

    def __init__(self, config, out_size):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, out_size, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x

class Embedder(nn.Module):
    """ base module to move from embedding space to data space  """

    def __init__(self, config, out_size):
        super().__init__()
        self.embed_now = nn.Embedding(10000, config.n_embd//2)
        self.embed_next = nn.Embedding(10000, config.n_embd//2)

    def forward(self, pos):
        pos_now = pos[:, 0, :]
        pos_next = pos[:, 1, :]
        return torch.cat(
            (self.embed_now(pos_now), self.embed_next(pos_next)),
            dim=-1,
        )

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

class GPT(nn.Module):

    def __init__(self, config, master_process=True):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.ModuleDict(dict(
                galaxy = Encoder(config, config.patch_size*config.patch_size*config.n_chan),
                spectra = Encoder(config, config.patch_size*config.patch_size*config.n_chan),
            )),
            wpe = nn.ModuleDict(dict(
                galaxy = Embedder(config, config.block_size),
                spectra = Embedder(config, config.block_size),
            )),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.ModuleDict(dict( 
            galaxy = Decoder(config, config.patch_size*config.patch_size*config.n_chan),
            spectra = Decoder(config, config.patch_size*config.patch_size*config.n_chan),
        ))
        # Tokens to give model knowledge of modality:
        self.token = nn.ParameterDict(dict(
            galaxy = nn.Parameter(torch.randn(1, 1, config.n_embd)),
            spectra = nn.Parameter(torch.randn(1, 1, config.n_embd)),
        ))
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
            modality = "galaxy" # dummy modality so we can take embedding count away
            n_params -= sum(p.numel() for p in self.transformer.wpe[modality].parameters())
            n_params -= sum(p.numel() for p in self.transformer.wte[modality].parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, spectra, targets=None, pos=None):
        device = images.device
        bb, tt_im, _ = images.size()
        _, tt_sp, _ = spectra.size()
        tt = tt_im + tt_sp
        assert tt <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if pos is None:
            pos = torch.arange(0, tt + 1, dtype=torch.long, device=device).unsqueeze(0) # shape (1, tt)
            pos = torch.stack((pos[..., :-1], pos[..., 1:]), dim=1)

        image_patches = self.transformer.wte["galaxy"](images)
        image_patches = image_patches + self.token["galaxy"]

        spect_patches = self.transformer.wte["spectra"](spectra)
        spect_patches = spect_patches + self.token["spectra"]

        # Calculate indices for each modality
        image_indices = range(0, image_patches.shape[1])
        spect_indices = range(image_patches.shape[1], spect_patches.shape[1] + image_patches.shape[1])

        tok_emb = torch.cat((image_patches, spect_patches), dim=1)

        # forward the GPT model itself
        #tok_emb = self.transformer.wte[modality](idx) # token embeddings of shape (bb, tt, n_embd)
        pos_emb = self.transformer.wpe["galaxy"](pos) # position embeddings of shape (1, tt, n_embd). For now we use a dummy "galaxy" position. Can improve later.
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        hidden_states = self.transformer.ln_f(x)
        image_states = hidden_states[:, image_indices]
        spect_states = hidden_states[:, spect_indices]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            image_logits = self.lm_head["galaxy"](image_states)
            spect_logits = self.lm_head["spectra"](spect_states)
            logits = torch.cat((image_logits, spect_logits), dim=1)
            loss = F.huber_loss(logits, targets)
        else:
            # TODO: get this working for arbitrary modalities
            # inference-time mini-optimization: only forward the lm_head on the very last position
            raise NotImplementedError
            #logits = self.lm_head[modality](x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            #loss = None

        return logits, loss

    def get_embeddings(self, idx, modality, pos=None, layer=None):
        device = idx.device
        b, t, ch = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if pos is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte[modality](idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe[modality](pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h[:layer]: # by default we take the penultimate layer as the embedding layer
            x = block(x)
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
    def generate_embeddings(self, idx, modality, average_type="mean", layer=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and get the embedding from the transformer model for that series.

        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.
        """
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        if average_type == "mean":
            # We only care about the average embedding
            weighted_embeddings = torch.mean(self.get_embeddings(idx_cond, modality, layer=layer), dim=1)
        elif average_type == "exp_decay":
            embeddings = self.get_embeddings(idx_cond, modality, layer=layer)
            weights = torch.logspace(0, -1, embeddings.shape[1], device=embeddings.device).unsqueeze(0).unsqueeze(-1)
            weighted_embeddings = torch.sum(weights*embeddings, dim=1)/torch.sum(embeddings, dim=1)
        elif average_type == "None":
            weighted_embeddings = self.get_embeddings(idx_cond, modality, layer=layer)
        else:
            raise NotImplementedError

        return weighted_embeddings
