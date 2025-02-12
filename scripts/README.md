# use the scripts

You will find in this folder an example script for pre-training (`train.py`).
You can combine this script with a config file in `../configs` for a different
training set up. By default we use the `galaxies` Huggingface repo of DESI
jpg images: `https://huggingface.co/datasets/Smith42/galaxies` to train
a 70M parameter model with causal attention.
