<p align="center">
    <img src="assets/emoji.png" alt="earthPT" width="150"/>
</p>

# astroPT: a Large Observation Model for astronomy

A simple repository for training astronomical large observation models. This
repository began its life as Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), and has been altered so that it
is usable for imagery data.  Within `train.py` you will find a ~300-line
boilerplate training loop and within `model.py` you will find a ~300-line GPT
model definition with an MLP tokeniser and a regressive loss.

Check out the discord for updates: [https://discord.gg/MNEVegvfJq](https://discord.gg/MNEVegvfJq)

## install

Dependencies:

- `pip install -r requirements.txt`

## results

AstroPT has been trained on 8.6M galaxy grz band `*.png` postage stamps 
downloaded from DESI-LS DR8 to see if neural scaling laws apply to galaxian
data (in other words, to see if `more galaxy data == more better model`).  
The results are promising, below I show our full training run validation losses
across a parameter sweep of `{1,5,12,21,89,309,830,2100}M` trainable parameters:

<p align="center">
    <img src="explore/scaling_xkcd.png" alt="scaling" width="512"/>
</p>

We also test our astroPT models on some scientifically-useful downstream tasks by
taking the models' penultimate layer outputs and finetuning linear probes to
predict emergent physical properties of the galaxies:

<p align="center">
    <img src="explore/downstream_xkcd.png" alt="scaling" width="512"/>
</p>

In the above pic, $M_g$ and $M_z$ are the absolute magnitudes (or brightness at
a fixed distance) of the galaxies, $g - r$ and $r - z$ are the differences
between the observations of different telescope filter bands, redshift is the
distance to the galaxies, sSFR is the total mass of new stars born each year in
the galaxies per total galaxy mass, and $M_{\*}$ is the total mass of stars within
the galaxies. "smooth?", "disc?", "artefact?", "edge on?" and "tight spiral?" are
morphological properties of the galaxies as described by citizen scientists.

## pretrained weights, and full galaxy dataset

Available on [HuggingFace 🤗 here](https://huggingface.co/Smith42/astroPT).

Dataset is also available on [HuggingFace 🔥](https://huggingface.co/datasets/Smith42/galaxies).
