<div align="center">
    
<img src="https://github.com/Smith42/astroPT/raw/main/assets/shoggoth_telescope_sticker_2.png" alt="astroPT_shoggoth" width="300"/>

<p></p>

[![PyPI](https://img.shields.io/pypi/v/astropt)](https://pypi.org/project/astropt/)
[![PyPI Downloads](https://static.pepy.tech/badge/astropt)](https://pepy.tech/projects/astropt) 
[![docs](https://app.readthedocs.org/projects/astropt/badge/)](https://astropt.readthedocs.io/)
[![License: AGPL-v3](https://img.shields.io/badge/License-AGPLv3-green.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
[![All Contributors](https://img.shields.io/badge/all_contributors-9-orange.svg?style=flat-square)](#contributors-)

[![ICML](https://img.shields.io/badge/AI4Science@ICML-2024---?logo=https%3A%2F%2Fneurips.cc%2Fstatic%2Fcore%2Fimg%2FNeurIPS-logo.svg&labelColor=68448B&color=b3b3b3)](https://openreview.net/forum?id=aOLuuLxqav)
[![arXiv](https://img.shields.io/badge/arXiv-2405.14930---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2405.14930)
[![arXiv](https://img.shields.io/badge/arXiv-2503.15312---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2503.15312)

[![arXiv](https://img.shields.io/badge/arXiv-2509.19453---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2509.19453)
[![arXiv](https://img.shields.io/badge/arXiv-2606.25610---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2606.25610)

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/Smith42/astroPT_v2.0)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/Smith42/galaxies)
</div>

# AstroPT: a Large Observation (foundation) Model for astronomy 🔭

Welcome to our simple repository for training astronomical large observation
models. This repository began its life as Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), and has been altered so that it
is usable for astronomical observation data.  Within `train.py` you will find a
~300-line boilerplate training loop and within `model.py` you will find a
~300-line GPT model definition with an MLP tokeniser and a regressive loss.

Check out the [UniverseTBD](https://universetbd.org/) Discord for updates:
[discord.gg/MNEVegvfJq](https://discord.gg/MNEVegvfJq)

Read the docs here: [astropt.readthedocs.io](https://astropt.readthedocs.io)

There is some [deep lore about our logo](https://doi.org/10.4000/12m9y)

# How does AstroPT work?

AstroPT is an autoregressive transformer under the hood.

Similarly to language models that predict the next word in a sentence, AstroPT processes sequences of astronomical data chunks to predict what comes next.

The intuition here is that this next-token-prediction task requires the model to internalise some understanding of the physical processes underlying the training data. 

This is just like how a text GPT needs to have some knowledge of geography to guess a country's capital given a description of that country, or some knowledge of coding to write compilable Fortran.

Below we can see this principle applied to a galaxy image, where we split the image into chunks and pass them into an AstroPT model:

<div align="center">
<img src="https://github.com/Smith42/astroPT/raw/main/assets/galaxy_im.png" alt="galaxy_im" width="25%"/>&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/Smith42/astroPT/raw/main/assets/apt.png" alt="astroPT_arch" width="70%"/>&nbsp;&nbsp;
</div>

Of course we can apply this next-token-prediction task across many modalities due to its flexibility.

Check out [our work on Euclid data](https://arxiv.org/abs/2503.15312) for an example, where we chain galaxy image tokens and spectral energy distribution data and pass them into a single, unified AstroPT model.

## Masked autoencoder (MAE) objective

As well as the default autoregressive objective, AstroPT can be pretrained with a
BERT-style masked autoencoder objective ([He et al. 2021](https://arxiv.org/abs/2111.06377),
[Devlin et al. 2019](https://arxiv.org/abs/1810.04805)). A fraction of the image
patches is replaced by a learnable mask token, the full patch sequence is
processed by the bidirectional encoder, and the masked patches are
reconstructed. Switch objectives with the `objective` config field (`"ar"` or
`"mae"`); MAE additionally requires bidirectional attention (`attn_type="full"`)
and uses AstroPT's existing learned (BERT-style) positional embeddings. The same
`scripts/train.py` runs both objectives — see `config/astropt_mae.py` for an
example. MAE currently supports a single image modality.

# I just want to run it! 🗣️

Okay I hear you! First you need to install the model:

## Install

You can install via pip from PyPI:
```bash
pip install astropt
```

Or if you install locally via a git clone, you can [uv](https://docs.astral.sh/uv/) install via:

```bash
git clone https://github.com/Smith42/astroPT.git
cd astroPT
uv sync
```

## Load a pre-trained model

To load and run a pre-trained AstroPT model from HuggingFace you can use the `load_astropt` function:

```python
from astropt.model_utils import load_astropt

model = load_astropt(
    repo_id="smith42/astropt_v2.0",
    path="astropt/095M",
    weights_filename="ckpt.pt",
)
model = model.to("cuda")
```

where `repo_id` is the HuggingFace repository ID, and `path` is the path within the repository that contains the AstroPT model checkpoint.

## Pre-trained models

Below are some pre-trained models you can load with the code snippet above.
Please make sure that you are using the correct version of AstroPT to load these!

| Survey | Modalities | AstroPT version | Model weights | Dataset | Paper |
| :----- | :--------- | :-------------- | :------------ | :------ | :---- |
| DESI Legacy Survey | JPG galaxy imagery | v1.0.0   | [AstroPT](https://huggingface.co/Smith42/astroPT) | [Galaxies Dataset](https://huggingface.co/datasets/Smith42/galaxies) | [arXiv:2405.14930](https://arxiv.org/abs/2405.14930) |
| Euclid | FITS VIS, NISP galaxy imagery and SED data | v1.0.2 | [AstroPT-Euclid](https://huggingface.co/collections/msiudek/astropt-euclid-67d061928ac0a447265ac8b8) | [Euclid Training Dataset](https://huggingface.co/datasets/msiudek/astroPT_euclid_training_dataset) | [arXiv:2503.15312](https://arxiv.org/abs/2503.15312) |
| DESI Legacy Survey | JPG galaxy imagery | v2.0.5   | [AstroPT v2.0](https://huggingface.co/Smith42/astroPT_v2.0) | [Galaxies Dataset v2.0](https://huggingface.co/datasets/Smith42/galaxies) | [arXiv:2405.14930](https://arxiv.org/abs/2405.14930) |

## Scripts for pre-training and processing data

Check out `scripts` for a collection of all the scripts we have used to get the
results in these papers, and `scripts/train.py` for an example boilerplate
script for pre-training your own AstroPT. `config` contains example user
configurations for pre-training.

AstroPT trains for roughly one epoch, so `train.py` can save intermediate
checkpoints by step count: set `num_checkpoints=N` to save `N` snapshots across
`[0, max_iters]` (always including the first/random-init and last/final step),
with `checkpoint_schedule` one of `"log"` (default; geometric, dense early —
good for probing how representations emerge over training), `"even"` (uniform),
or `"manual"` (use the explicit `checkpoint_steps` list, e.g.
`--checkpoint_steps=[0,512,4096,30000]`). This is independent of the best-val
`ckpt.pt`; each checkpoint also stores optimizer state, so budget disk
accordingly.

`scripts/linear_probe.py` has an example script for inferring embeddings from a 
pre-trained model and  running a finetuning routine on them 🌝.

And finally `scripts/finetune.py` has an example LoRA finetune routine.

## Multi-GPU streaming

When streaming the dataset from HuggingFace under DDP, `train.py` shards the
stream across ranks with `split_dataset_by_node` so each GPU sees disjoint data
(otherwise every rank replays the same stream and you get no data-throughput
scaling), and applies a buffered `shuffle` (size `shuffle_buffer_size`) to the
training stream.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RJ-Roberts"><img src="https://avatars.githubusercontent.com/u/131991163?v=4?s=100" width="100px;" alt="Ryan Roberts"/><br /><sub><b>Ryan Roberts</b></sub></a><br /><a href="https://github.com/Smith42/astroPT/commits?author=RJ-Roberts" title="Code">💻</a> <a href="#ideas-RJ-Roberts" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-RJ-Roberts" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://mjjsmith.com/"><img src="https://avatars.githubusercontent.com/u/8194280?v=4?s=100" width="100px;" alt="Mike Smith"/><br /><sub><b>Mike Smith</b></sub></a><br /><a href="https://github.com/Smith42/astroPT/commits?author=Smith42" title="Code">💻</a> <a href="#ideas-Smith42" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-Smith42" title="Content">🖋</a> <a href="#data-Smith42" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mhuertascompany"><img src="https://avatars.githubusercontent.com/u/22987973?v=4?s=100" width="100px;" alt="mhuertascompany"/><br /><sub><b>mhuertascompany</b></sub></a><br /><a href="#ideas-mhuertascompany" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-mhuertascompany" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/msiudek"><img src="https://avatars.githubusercontent.com/u/53626980?v=4?s=100" width="100px;" alt="Malgorzata Siudek"/><br /><sub><b>Malgorzata Siudek</b></sub></a><br /><a href="#ideas-msiudek" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-msiudek" title="Content">🖋</a> <a href="https://github.com/Smith42/astroPT/commits?author=msiudek" title="Code">💻</a> <a href="#data-msiudek" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gimarso"><img src="https://avatars.githubusercontent.com/u/52239656?v=4?s=100" width="100px;" alt="gimarso"/><br /><sub><b>gimarso</b></sub></a><br /><a href="#ideas-gimarso" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Smith42/astroPT/commits?author=gimarso" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/victoralonsorodriguez"><img src="https://avatars.githubusercontent.com/u/69580371?v=4?s=100" width="100px;" alt="Víctor Alonso"/><br /><sub><b>Víctor Alonso</b></sub></a><br /><a href="https://github.com/Smith42/astroPT/issues?q=author%3Avictoralonsorodriguez" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ashodkh"><img src="https://avatars.githubusercontent.com/u/81383507?v=4?s=100" width="100px;" alt="Ashod Khederlarian"/><br /><sub><b>Ashod Khederlarian</b></sub></a><br /><a href="https://github.com/Smith42/astroPT/commits?author=ashodkh" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sogolsnj"><img src="https://avatars.githubusercontent.com/u/88580847?v=4?s=100" width="100px;" alt="SogolSanjaripour"/><br /><sub><b>SogolSanjaripour</b></sub></a><br /><a href="https://github.com/Smith42/astroPT/commits?author=sogolsnj" title="Code">💻</a> <a href="#ideas-sogolsnj" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://ksd3.github.io"><img src="https://avatars.githubusercontent.com/u/89213519?v=4?s=100" width="100px;" alt="ksd3"/><br /><sub><b>ksd3</b></sub></a><br /><a href="#ideas-ksd3" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Smith42/astroPT/commits?author=ksd3" title="Code">💻</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
