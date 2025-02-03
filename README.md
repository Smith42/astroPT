<div align="center">
    
<img src="https://github.com/Smith42/astroPT/raw/main/assets/shoggoth_telescope_sticker_2.png" alt="astroPT_shoggoth" width="300"/>

[![ICML](https://img.shields.io/badge/AI4Science@ICML-2024---?logo=https%3A%2F%2Fneurips.cc%2Fstatic%2Fcore%2Fimg%2FNeurIPS-logo.svg&labelColor=68448B&color=b3b3b3)](https://openreview.net/forum?id=aOLuuLxqav)
[![arXiv](https://img.shields.io/badge/arXiv-2405.14930---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/2405.14930)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
</div>

# astroPT: a Large Observation Model for astronomy 🔭

Welcome to our simple repository for training astronomical large observation
models. This repository began its life as Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), and has been altered so that it
is usable for imagery data.  Within `train.py` you will find a ~300-line
boilerplate training loop and within `model.py` you will find a ~300-line GPT
model definition with an MLP tokeniser and a regressive loss.

Check out the [UniverseTBD](https://universetbd.org/) Discord for updates:
[https://discord.gg/MNEVegvfJq](https://discord.gg/MNEVegvfJq)

## install

You can install via pip from PyPI:
```bash
pip install your-package-name
```

Or if you install locally, you can get the dependencies via:

```bash
pip install -r requirements.txt
```

## results

AstroPT v1.0.0 has been trained on 8.6M galaxy grz band `*.png` postage stamps 
downloaded from DESI-LS DR8 to see if neural scaling laws apply to galaxian
data (in other words, to see if `more galaxy data == more better model`).  
We tried to make the astroPT model as simple as possible so that other
modalities can be easily folded in. We also choose to use a causally trained
autoregressive transformer model as our backbone so that our work can more
easily integrate the wider deep learning FOSS community.

Our pretraining task is feeding in our galaxy images patch-by-patch and
predicting the next patch in our galaxy patch sequence. We follow ViT
and define a patch as a 16 by 16 pixel square, and feed the galaxy patches
in a spiral order:

<p align="center">
    <img src="https://github.com/Smith42/astroPT/raw/main/explore/galaxy.png" alt="galaxy" width="128"/>
</p>

The trained model results are promising -- below we show our full training run
validation losses across a parameter sweep of `{1,5,12,21,89,309,830,2100}M`
trainable parameters:

<p align="center">
    <img src="https://github.com/Smith42/astroPT/raw/main/explore/scaling_xkcd.png" alt="scaling" width="512"/>
</p>

We also test our astroPT models on some scientifically-useful downstream tasks by
taking the models' penultimate layer outputs and finetuning linear probes to
predict emergent physical properties of the galaxies:

<p align="center">
    <img src="https://github.com/Smith42/astroPT/raw/main/explore/downstream_xkcd.png" alt="downstream" width="512"/>
</p>

In the above pic, $M_g$ and $M_z$ are the absolute magnitudes (or brightness at
a fixed distance) of the galaxies, $g - r$ and $r - z$ are the differences
between the observations of different telescope filter bands, redshift is the
distance to the galaxies, sSFR is the total mass of new stars born each year in
the galaxies per total galaxy mass, and $M_{\*}$ is the total mass of stars within
the galaxies. "smooth?", "disc?", "artefact?", "edge on?" and "tight spiral?" are
morphological properties of the galaxies as described by citizen scientists.

The cool thing to take away from these plots is that the surrogate task loss
(predicting the next patch in a sequence of ViT-like galaxy image patches)
is correlated with astronomically "useful" downstream tasks 🤯🚀.

Finally, check out our UMAP projection of astroPT-87M's penultimate layer
outputs of our validation set. We colour each point with an emergent physical
galaxy property described above. The structure suggests that the model has
learnt some knowledge about physics simply from our next-token prediction
pretraining task!

<p align="center">
    <img src="https://github.com/Smith42/astroPT/raw/main/explore/hexbin_xkcd.png" alt="hexbin" width="512"/>
</p>

## pretrained weights, and full galaxy dataset

Check out the paper here: [https://arxiv.org/abs/2405.14930](https://arxiv.org/abs/2405.14930).

We of course release all our model weights checkpointed across our full training runs on [HuggingFace 🤗 here](https://huggingface.co/Smith42/astroPT).

We also release our full dataset and galaxy metadata on [HuggingFace 🔥](https://huggingface.co/datasets/Smith42/galaxies).

## contributors

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
