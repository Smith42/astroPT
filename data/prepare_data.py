"""
This file translates galaxy imagery to sequences ready for our astroPT model to train on.

2024-01 mike.smith@aspiaspace.com
"""
import numpy as np
import h5py as h5
import re
import sys
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

def get_fits(fname):
    """
    Convert fits file to numpy array.
    """

    # TODO get this working with the fits files
    with fits.File(fname[0], "r") as f:
        cs = np.stack([f[fname[0].name[:4]][ch] 
            for ch in [ "g", "r", "z", ]])

    # TODO figure out normalisation
    return (cs/1000)*2-1

if __name__ == "__main__":
    grid = int(sys.argv[1])
    tile = f"TL{grid:02d}"

    for split in ["train", "test"]:
        fnames = np.array(sorted(Path(f"./galaxies/").glob("*.fits")))

        splitpoint = int(len(fnames)*0.8) 
        if split == "train":
            fnames = fnames[:splitpoint]
        else:
            fnames = fnames[splitpoint:]

        print(f"{tile}: processing {len(fnames)} frames")

        # TODO how do we want to structure bigar? 
        # We will probably want this to be [16*16*len(fnames), 1024*3]
        bigar = np.zeros((len(fnames)-1, 512, 512, 3))
        for i, fname, next_fname in tqdm(zip(
                                      range(len(fnames)-1), 
                                      fnames,
                                      fnames[1:]),
                                      total=len(fnames)-1
                                    ):
            bigar[:, i] = np.swapaxes(get_h5(fname, next_fname).reshape(18, 1024*1024), 0, 1)

        np.save(f"./TL_EOPT/{tile}_{split}.npy", bigar.astype(np.float16))
