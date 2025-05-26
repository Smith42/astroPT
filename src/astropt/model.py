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

import inspect
import math
import random
from dataclasses import dataclass

import loralib as lora
import torch
import torch.nn as nn

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
        or_masks,
    )

    flex_attention_avail = True
except ImportError:
    print(
        "WARNING: only causal attention is available. Flex Attention requires PyTorch >= 2.6"
    )
    flex_attention_avail = False
from torch.nn import functional as F


@dataclass
class ModalityConfig:
    """Configuration for a single modality"""

    name: str
    input_size: int
    pos_input_size: int
    patch_size: int
    embed_pos: bool
    loss_weight: float = 1.0


class ModalityRegistry:
    """Central registry for model modalities"""

    def __init__(self, modalities):
        self.modalities = {m.name: m for m in modalities}

    def get_config(self, name):
        """Get configuration for a specific modality"""
        return self.modalities[name]

    def names(self):
        """Get names of modalities"""
        return sorted(self.modalities.keys())

    def generate_sequence(self, num_sequences=1, shuf=False):
        """Generate a modality sequence from available modalities"""
        if shuf:
            return random.sample(self.names(), len(self.names()))
        return self.names()


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

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
        if hasattr(config, "lora_r") and config.lora_r > 0:
            self.c_attn = lora.Linear(
                config.n_embd, 3 * config.n_embd, r=config.lora_r, bias=config.bias
            )
        else:
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
            self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
            if not self.flash:
                print(
                    "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
                )
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer(
                    "bias",
                    torch.tril(torch.ones(config.block_size, config.block_size)).view(
                        1, 1, config.block_size, config.block_size
                    ),
                )
        elif self.attn_type == "prefix" and flex_attention_avail:
            # flex attention also make GPU brrrr for non-causal masking, only available with DDP for PyTorch >= 2.6
            # need to compile flex attention for performance!
            self.flex_attention = torch.compile(flex_attention)
        else:
            raise NotImplementedError(
                "Attention type must be one of 'causal' or 'prefix'. Prefix requires PyTorch >= 2.6."
            )

    def forward(self, x, block_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.attn_type == "causal":
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        elif self.attn_type == "prefix":
            y = self.flex_attention(q, k, v, block_mask=block_mask)
        else:
            raise NotImplementedError(
                "Attention type must be one of 'causal' or 'prefix'. Prefix requires PyTorch >= 2.6."
            )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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


class TaskHead(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(config.n_embd // 2, output_dim),
        )

    def forward(self, x):
        # Global average pooling over sequence length
        x = x.mean(dim=1)  # (batch, n_embd)
        x = self.ln(x)
        x = self.head(x)
        return x


class Encoder(nn.Module):
    """base module to move from data space to embedding space"""

    def __init__(self, config, in_size):
        super().__init__()
        self.c_fc = nn.Linear(in_size, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x


class Decoder(nn.Module):
    """base module to move from embedding space to data space"""

    def __init__(self, config, out_size):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, out_size, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x


class Embedder(nn.Module):
    """base module to move from embedding space to data space"""

    def __init__(self, config):
        super().__init__()
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, pos):
        return self.wpe(pos)


@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_chan: int = 1
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attn_type: str = "causal"  # causal or prefix
    # LoRA params
    lora_r: int = 0  # rank, 0 disables LoRA
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    modalities: list[ModalityConfig] = None


class GPT(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        modality_registry: ModalityRegistry,
        master_process=True,
    ):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        self.modality_registry = modality_registry

        # create encoders and decoders
        encoders = {}
        decoders = {}
        embedders = {}
        for name, mod_config in modality_registry.modalities.items():
            encoders[name] = Encoder(config, mod_config.input_size)
            if mod_config.embed_pos:
                embedders[name] = Embedder(config)
            else:
                embedders[name] = Encoder(config, mod_config.pos_input_size)
            decoders[name] = Decoder(config, mod_config.input_size)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.ModuleDict(encoders),
                wpe=nn.ModuleDict(embedders),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.ModuleDict(decoders)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # TODO rethink weight tying
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # optional task head for finetuning
        self.task_head = None
        if hasattr(config, "output_dim"):
            self.task_head = TaskHead(config, config.output_dim)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        self.master_process = master_process
        if self.master_process:
            print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, targets=None, prefix_len=None, target_modality=None, attention_mask=None):
        tt = sum(v.size(1) for k, v in inputs.items() if k.endswith("_positions"))
        assert tt <= self.config.block_size, (
            f"Cannot forward sequence of length {tt}, block size is only {self.config.block_size}"
        )
        if self.config.attn_type == "prefix":
            # TODO we need to make sure that the prefix hyperparameters are tuned well:
            if prefix_len is None:
                # if we don't pass a prefix length assume we want it sampled at random
                # TODO do we want this to be an eval mode switch?
                prefix_len = random.randrange(self.config.block_size - 1)
            prefix_lm_mask = generate_prefix_lm_mask(prefix_len)
            block_mask = create_block_mask(
                prefix_lm_mask,
                None,
                None,
                self.config.block_size,
                self.config.block_size,
            )
        else:
            block_mask = None

        # forward the GPT model itself
        embeddings = []
        pos_embeddings = []
        for mod_name in self.modality_registry.names():
            input_tensor = inputs[mod_name]
            embeddings.append(self.transformer.wte[mod_name](input_tensor))
            pos = inputs[mod_name + "_positions"]
            pos_embeddings.append(self.transformer.wpe[mod_name](pos))
        tok_emb = torch.cat(embeddings, dim=1)
        pos_emb = torch.cat(pos_embeddings, dim=1)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, block_mask=block_mask)
        x = self.transformer.ln_f(x)

        outputs = {}
        current_idx = 0

        if target_modality is not None:
            # continue sequence if target modality is in inputs
            if target_modality in inputs:
                for mod_name in self.modality_registry.names():
                    input_tensor = inputs[mod_name]
                    if mod_name == target_modality:
                        seq_len = input_tensor.size(1)
                        hidden_state = x[:, current_idx : current_idx + seq_len]
                        outputs[mod_name] = self.lm_head[mod_name](hidden_state)
                    current_idx += input_tensor.size(1)
            # target modality not in inputs so start a new sequence
            else:
                hidden_state = x[:, -1:, :]
                outputs[target_modality] = self.lm_head[target_modality](hidden_state)

        for ii, mod_name in enumerate(self.modality_registry.names()):
            input_tensor = inputs[mod_name]
            seq_len = input_tensor.size(1)
            # If we have more than one mode, the last value of the past modes
            # are used to prompt the next mode gen:
            if ii == 0 and len(self.modality_registry.names()) > 1:
                seq_len = seq_len - 1
            hidden_state = x[:, current_idx : current_idx + seq_len]
            outputs[mod_name] = self.lm_head[mod_name](hidden_state)
            current_idx += seq_len

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            current_idx = 0
            loss = 0
            for mod_name in self.modality_registry.names():
                target = targets[mod_name]
                seq_len = target.size(1)
                mod_config = self.modality_registry.get_config(mod_name)
                pred = outputs[mod_name]
                if self.config.attn_type == "prefix":
                    # TODO fix this and debug
                    raise NotImplementedError(
                        "Prefix attention not implemented for multimodal model"
                    )
                    ## if we have prefix attention on we only want to
                    ## backprop through tokens where our model cannot
                    ## look ahead! so we mask the loss to prefix_len
                    # if current_idx + seq_len <= prefix_len:
                    #    # entire modality within prefix so skip
                    #    continue
                    # elif current_idx >= prefix_len:
                    #    # entire modality beyond prefix so backprop
                    #    loss += F.huber_loss(pred, target) * mod_config.loss_weight
                    # else:
                    #    # some of modality within prefix so mask and bp
                    #    prefix_mask = torch.ones_like(target, dtype=torch.bool)
                    #    prefix_sublen = prefix_len - current_idx
                    #    prefix_mask[:, :prefix_sublen] = False
                elif attention_mask is not None:
                    # Extract attention mask for this modality
                    mod_mask = attention_mask[:, current_idx:current_idx + seq_len]

                    unmasked_loss = F.huber_loss(pred, target, reduction="none")
                    mask = mod_mask.unsqueeze(-1)
                    masked_loss = (unmasked_loss * mask).sum() / mask.sum()
                    loss += masked_loss * mod_config.loss_weight
                else:
                    loss += F.huber_loss(pred, target) * mod_config.loss_weight
                current_idx += seq_len
            loss /= len(self.modality_registry.names())
        else:
            loss = None

        return outputs, loss

    def get_embeddings(self, inputs, draw_from_centre=True, prefix_len=None):
        """
        Get embeddings from AstroPT.

        Args:
            inputs: dict of tensors with modality names as keys
            draw_from_centre = get embedding from centre from model not from penultimate layer
            prefix_len: optional prefix length to consider

        Returns:
            dictionary of embeddings for each modality
        """
        tt = sum(v.size(1) for k, v in inputs.items() if k.endswith("_positions"))
        assert tt <= self.config.block_size, (
            f"Cannot forward sequence of length {tt}, block size is only {self.config.block_size}"
        )

        # generate token embeddings per modality
        embeddings = []
        pos_embeddings = []
        for mod_name in self.modality_registry.names():
            input_tensor = inputs[mod_name]
            embeddings.append(self.transformer.wte[mod_name](input_tensor))
            pos = inputs[mod_name + "_positions"]
            pos_embeddings.append(self.transformer.wpe[mod_name](pos))
        tok_emb = torch.cat(embeddings, dim=1)
        pos_emb = torch.cat(pos_embeddings, dim=1)
        x = self.transformer.drop(tok_emb + pos_emb)

        for i, block in enumerate(
            self.transformer.h
        ):  # by default we take the penultimate layer as the embedding layer
            x = block(x)
            if draw_from_centre and i == len(self.transformer.h) // 2:
                centre_embeddings = x

        if not draw_from_centre:
            embeddings_out = self.transformer.ln_f(x)
        else:
            embeddings_out = centre_embeddings

        # split embeddings by modality
        result = {}
        current_idx = 0
        for mod_name in self.modality_registry.names():
            input_tensor = inputs[mod_name]
            seq_len = input_tensor.size(1)
            result[mod_name] = embeddings_out[:, current_idx : current_idx + seq_len]
            current_idx += seq_len

        return result

    def get_task_prediction(self, inputs, prefix_len=None, targets=None):
        """Forward pass for task prediction during finetuning"""
        if self.task_head is None:
            raise ValueError(
                "Model not configured for task prediction. Set config.output_dim"
            )
        tt = sum(v.size(1) for k, v in inputs.items() if k.endswith("_positions"))
        assert tt <= self.config.block_size, (
            f"Cannot forward sequence of length {tt}, block size is only {self.config.block_size}"
        )

        # forward the GPT model itself
        # generate token embeddings per modality
        embeddings = []
        pos_embeddings = []
        for mod_name in self.modality_registry.names():
            input_tensor = inputs[mod_name]
            embeddings.append(self.transformer.wte[mod_name](input_tensor))
            pos = inputs[mod_name + "_positions"]
            pos_embeddings.append(self.transformer.wpe[mod_name](pos))
        tok_emb = torch.cat(embeddings, dim=1)
        pos_emb = torch.cat(pos_embeddings, dim=1)
        x = self.transformer.drop(tok_emb + pos_emb)
        for ii, block in enumerate(self.transformer.h):
            x = block(x)
            if ii == len(self.transformer.h) // 2:  # Take features from middle layer
                break

        outputs = self.task_head(x)
        return (
            (outputs, F.huber_loss(outputs, targets))
            if targets is not None
            else outputs
        )

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

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
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        if self.master_process:
            print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, inputs, new_tokens, target_modality, temperature=0.0):
        """
        Take a conditioning sequence for each modality and generate new tokens for the target modality.

        Args:
            inputs: dict of tokens with modality names as keys
            new_tokens: number of new tokens to generate
            target_modality: modality to generate tokens for
            temperature: temperature for sampling (0.0 = deterministic)

        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for ii in range(new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            for mod_name in self.modality_registry.names():
                if inputs[mod_name].size(1) > self.config.block_size:
                    inputs[mod_name] = inputs[mod_name][:, -self.config.block_size :]

            # forward the model to get the logits for the index in the sequence
            outputs, _ = self(inputs, target_modality=target_modality)
            next_token = outputs[target_modality][:, -1:, :]
            next_token = next_token + torch.randn_like(next_token) * temperature

            if target_modality not in inputs:
                inputs[target_modality] = next_token
            else:
                inputs[target_modality] = torch.cat(
                    [inputs[target_modality], next_token], dim=1
                )

        return inputs

    @torch.no_grad()
    def generate_embeddings(self, inputs, reduction="mean"):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and get the embedding from the transformer model for that series.

        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.
        """
        embeddings_dict = self.get_embeddings(inputs)
        result = {}

        # apply reduction for each modality
        for mod_name, embeddings in embeddings_dict.items():
            if reduction == "mean":
                result[mod_name] = torch.mean(embeddings, dim=1)
            elif reduction == "exp_decay":
                weights = (
                    torch.logspace(0, -1, embeddings.shape[1], device=embeddings.device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                result[mod_name] = torch.sum(weights * embeddings, dim=1) / torch.sum(
                    weights, dim=1
                )
            elif reduction == "last":
                result[mod_name] = embeddings[:, -1, :]
            elif reduction == "none":
                result[mod_name] = embeddings
            else:
                raise NotImplementedError(
                    f"Reduction method '{reduction}' not implemented"
                )

        return result
