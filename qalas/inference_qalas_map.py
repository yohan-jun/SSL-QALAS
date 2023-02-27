"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
import pathlib
from collections import defaultdict
from pathlib import Path

import fastmri
import fastmri.data.transforms_qalas as T
import numpy as np
import requests
import torch
import pytorch_lightning as pl
from fastmri.data import SliceDatasetQALAS
from fastmri.models import QALAS_MAP
from fastmri.pl_modules import QALAS_MAPModule
from tqdm import tqdm

MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def run_model(batch, model, device):
    crop_size = batch.crop_size

    output_t1, output_t2, output_pd, output_ie, output_b1, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            model(batch.masked_kspace_acq1.to(device), batch.masked_kspace_acq2.to(device), batch.masked_kspace_acq3.to(device), batch.masked_kspace_acq4.to(device), batch.masked_kspace_acq5.to(device), \
                batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                batch.coil_sens, batch.b1, batch.ie, \
                batch.max_value_t1.to(device), batch.max_value_t2.to(device), batch.max_value_pd.to(device), batch.num_low_frequencies.to(device))

    # detect FLAIR 203
    if output_t1.shape[-1] < crop_size[1]:
        crop_size = (output_t1.shape[-1], output_t1.shape[-1])

    output_t1 = T.center_crop(output_t1, crop_size)[0]
    output_t2 = T.center_crop(output_t2, crop_size)[0]
    output_pd = T.center_crop(output_pd, crop_size)[0]
    output_ie = T.center_crop(output_ie, crop_size)[0]
    output_b1 = T.center_crop(output_b1, crop_size)[0]

    output_t1 = output_t1 * batch.mask_brain.to(device)
    output_t2 = output_t2 * batch.mask_brain.to(device)
    output_pd = output_pd * batch.mask_brain.to(device)
    output_ie = output_ie * batch.mask_brain.to(device)
    output_b1 = output_b1 * batch.mask_brain.to(device)

    return output_t1, output_t2, output_pd, output_ie, output_b1, int(batch.slice_num[0]), batch.fname[0]

def load_model(
    module_class: pl.LightningModule,
    fname: pathlib.Path,
):
    print(f"loading model from {fname}")
    # checkpoint = torch.load(fname, map_location=torch.device("cpu"))
    checkpoint = torch.load(fname, map_location=torch.device("cuda"))

    # Initialise model with stored params
    module = module_class(**checkpoint["hyper_parameters"])

    # Load stored weights: this will error if the keys don't match the model weights, which will happen
    #  when we are loading a VarNet instead of an AdaptiveVarNet or vice-versa.
    module.load_state_dict(checkpoint["state_dict"])

    return module

def run_inference(challenge, state_dict_file, data_path, output_path, device):
    # model = QALAS_MAP()

    model = load_model(QALAS_MAPModule, state_dict_file)

    # model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()

    # data loader setup
    data_transform = T.QALASDataTransform()
    dataset = SliceDatasetQALAS(
        root=data_path, transform=data_transform, challenge="multicoil"
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs_t1 = defaultdict(list)
    outputs_t2 = defaultdict(list)
    outputs_pd = defaultdict(list)
    outputs_ie = defaultdict(list)
    outputs_b1 = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output_t1, output_t2, output_pd, output_ie, output_b1, slice_num, fname = run_model(batch, model, device)
        outputs_t1[fname].append((slice_num, np.transpose(np.squeeze(output_t1.cpu()),(1,0))))
        outputs_t2[fname].append((slice_num, np.transpose(np.squeeze(output_t2.cpu()),(1,0))))
        outputs_pd[fname].append((slice_num, np.transpose(np.squeeze(output_pd.cpu()),(1,0))))
        outputs_ie[fname].append((slice_num, np.transpose(np.squeeze(output_ie.cpu()),(1,0))))
        outputs_b1[fname].append((slice_num, np.transpose(np.squeeze(output_b1.cpu()),(1,0))))

    # save outputs
    for fname in outputs_t1:
        outputs_t1[fname] = np.stack([out for _, out in sorted(outputs_t1[fname])])
        outputs_t2[fname] = np.stack([out for _, out in sorted(outputs_t2[fname])])
        outputs_pd[fname] = np.stack([out for _, out in sorted(outputs_pd[fname])])
        outputs_ie[fname] = np.stack([out for _, out in sorted(outputs_ie[fname])])
        outputs_b1[fname] = np.stack([out for _, out in sorted(outputs_b1[fname])])

    fastmri.save_reconstructions_qalas(outputs_t1, outputs_t2, outputs_pd, outputs_ie, outputs_b1, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="multicoil",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )
