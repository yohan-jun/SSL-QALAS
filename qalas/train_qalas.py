"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms_qalas import QALASDataTransform
from fastmri.pl_modules import FastMriDataModuleQALAS, QALAS_MAPModule

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    if args.mask_type == 'qalas':
        mask = create_mask_for_mask_type(
        'qalas', args.center_fractions, args.accelerations
        )
    else:
        mask = create_mask_for_mask_type(
            args.mask_type, args.center_fractions, args.accelerations
        )
    # use random masks for train transform, fixed masks for val transform
    
    # If load masks in 'subsample.py', please use following two lines
    # train_transform = QALASDataTransform(mask_func_acq1=mask, mask_func_acq2=mask, mask_func_acq3=mask, mask_func_acq4=mask, mask_func_acq5=mask, use_seed=False)
    # val_transform = QALASDataTransform(mask_func_acq1=mask, mask_func_acq2=mask, mask_func_acq3=mask, mask_func_acq4=mask, mask_func_acq5=mask)

    # If masks are saved in .h5 file, please use following two lines
    train_transform = QALASDataTransform()
    val_transform = QALASDataTransform()
    test_transform = QALASDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModuleQALAS(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = QALAS_MAPModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        maps_chans=args.maps_chans,
        maps_layers=args.maps_layers,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, gpus=[0], log_every_n_steps=1)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("brain_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "qalas_log"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="equispaced_fraction",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.005],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModuleQALAS.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="qalas",  # "qalas"
        challenge="multicoil",  # only multicoil implemented for QALAS
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = QALAS_MAPModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=1,  # number of unrolled iterations
        pools=3,  # number of pooling layers for U-Net (default: 3)
        chans=64,  # number of top-level channels for U-Net (default: 64)
        maps_chans=64,  # number of channels for mapping est. CNN (defalut: 64)
        maps_layers=5,  # number of layers for mapping est. CNN (default: 5)
        lr=0.001,  # Adam learning rate (default: 0.001)
        lr_step_size=1000,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=500,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory

    # if args.resume_from_checkpoint is None:
    #     ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
    #     if ckpt_list:
    #         args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
