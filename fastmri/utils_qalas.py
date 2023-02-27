"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np


def save_reconstructions_qalas(reconstructions_t1: Dict[str, np.ndarray], reconstructions_t2: Dict[str, np.ndarray], \
                            reconstructions_pd: Dict[str, np.ndarray], reconstructions_ie: Dict[str, np.ndarray], \
                            reconstructions_b1: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons_t1 in reconstructions_t1.items():
        recons_t2 = reconstructions_t2[fname]
        recons_pd = reconstructions_pd[fname]
        recons_ie = reconstructions_ie[fname]
        recons_b1 = reconstructions_b1[fname]
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction_t1", data=recons_t1)
            hf.create_dataset("reconstruction_t2", data=recons_t2)
            hf.create_dataset("reconstruction_pd", data=recons_pd)
            hf.create_dataset("reconstruction_ie", data=recons_ie)
            hf.create_dataset("reconstruction_b1", data=recons_b1)

def save_reconstructions_qalas_forward(reconstructions_img1: Dict[str, np.ndarray], reconstructions_img2: Dict[str, np.ndarray], \
                            reconstructions_img3: Dict[str, np.ndarray], reconstructions_img4: Dict[str, np.ndarray], \
                            reconstructions_img5: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons_img1 in reconstructions_img1.items():
        recons_img2 = reconstructions_img2[fname]
        recons_img3 = reconstructions_img3[fname]
        recons_img4 = reconstructions_img4[fname]
        recons_img5 = reconstructions_img5[fname]
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction_img1", data=recons_img1)
            hf.create_dataset("reconstruction_img2", data=recons_img2)
            hf.create_dataset("reconstruction_img3", data=recons_img3)
            hf.create_dataset("reconstruction_img4", data=recons_img4)
            hf.create_dataset("reconstruction_img5", data=recons_img5)

def convert_fnames_to_v2(path: Path):
    """
    Converts filenames to conform to `v2` standard for knee data.

    For a file with name file1000.h5 in `path`, this script simply renames it
    to file1000_v2.h5. This is for submission to the public knee leaderboards.

    Args:
        path: Path with files to be renamed.
    """
    if not path.exists():
        raise ValueError("Path does not exist")

    for fname in path.glob("*.h5"):
        if not fname.name[-6:] == "_v2.h5":
            fname.rename(path / (fname.stem + "_v2.h5"))
