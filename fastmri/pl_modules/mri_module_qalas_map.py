"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import fastmri
import numpy as np
import pytorch_lightning as pl
import torch
from fastmri import evaluate
from torchmetrics.metric import Metric


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModuleQALAS_MAP(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 4):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value_t1",
            "max_value_t2",
            "max_value_pd",
            "output_t1",
            "output_t2",
            "output_pd",
            "output_ie",
            "output_b1",
            "output_img1",
            "output_img2",
            "output_img3",
            "output_img4",
            "output_img5",
            "target_t1",
            "target_t2",
            "target_pd",
            "target_img1",
            "target_img2",
            "target_img3",
            "target_img4",
            "target_img5",
            "val_loss_t1",
            "val_loss_t2",
            "val_loss_pd",
            "val_loss_img1",
            "val_loss_img2",
            "val_loss_img3",
            "val_loss_img4",
            "val_loss_img5",
            # "val_loss_img1_tv",
            # "val_loss_img2_tv",
            # "val_loss_img3_tv",
            # "val_loss_img4_tv",
            # "val_loss_img5_tv",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output_t1"].ndim == 2:
            val_logs["output_t1"] = val_logs["output_t1"].unsqueeze(0)
        elif val_logs["output_t1"].ndim != 3:
            raise RuntimeError("Unexpected output_t1 size from validation_step.")
        if val_logs["output_t2"].ndim == 2:
            val_logs["output_t2"] = val_logs["output_t2"].unsqueeze(0)
        elif val_logs["output_t2"].ndim != 3:
            raise RuntimeError("Unexpected output_t2 size from validation_step.")
        if val_logs["output_pd"].ndim == 2:
            val_logs["output_pd"] = val_logs["output_pd"].unsqueeze(0)
        elif val_logs["output_pd"].ndim != 3:
            raise RuntimeError("Unexpected output_pd size from validation_step.")
        if val_logs["output_ie"].ndim == 2:
            val_logs["output_ie"] = val_logs["output_ie"].unsqueeze(0)
        elif val_logs["output_ie"].ndim != 3:
            raise RuntimeError("Unexpected output_ie size from validation_step.")
        if val_logs["output_b1"].ndim == 2:
            val_logs["output_b1"] = val_logs["output_b1"].unsqueeze(0)
        elif val_logs["output_b1"].ndim != 3:
            raise RuntimeError("Unexpected output_b1 size from validation_step.")
        if val_logs["output_img1"].ndim == 2:
            val_logs["output_img1"] = val_logs["output_img1"].unsqueeze(0)
        elif val_logs["output_img1"].ndim != 3:
            raise RuntimeError("Unexpected output_img1 size from validation_step.")
        if val_logs["output_img2"].ndim == 2:
            val_logs["output_img2"] = val_logs["output_img2"].unsqueeze(0)
        elif val_logs["output_img2"].ndim != 3:
            raise RuntimeError("Unexpected output_img2 size from validation_step.")
        if val_logs["output_img3"].ndim == 2:
            val_logs["output_img3"] = val_logs["output_img3"].unsqueeze(0)
        elif val_logs["output_img3"].ndim != 3:
            raise RuntimeError("Unexpected output_img3 size from validation_step.")
        if val_logs["output_img4"].ndim == 2:
            val_logs["output_img4"] = val_logs["output_img4"].unsqueeze(0)
        elif val_logs["output_img4"].ndim != 3:
            raise RuntimeError("Unexpected output_img4 size from validation_step.")
        if val_logs["output_img5"].ndim == 2:
            val_logs["output_img5"] = val_logs["output_img5"].unsqueeze(0)
        elif val_logs["output_img5"].ndim != 3:
            raise RuntimeError("Unexpected output_img5 size from validation_step.")
        if val_logs["target_t1"].ndim == 2:
            val_logs["target_t1"] = val_logs["target_t1"].unsqueeze(0)
        elif val_logs["target_t1"].ndim != 3:
            raise RuntimeError("Unexpected target_t1 size from validation_step.")
        if val_logs["target_t2"].ndim == 2:
            val_logs["target_t2"] = val_logs["target_t2"].unsqueeze(0)
        elif val_logs["target_t2"].ndim != 3:
            raise RuntimeError("Unexpected target_t2 size from validation_step.")
        if val_logs["target_pd"].ndim == 2:
            val_logs["target_pd"] = val_logs["target_pd"].unsqueeze(0)
        elif val_logs["target_pd"].ndim != 3:
            raise RuntimeError("Unexpected target_pd size from validation_step.")
        if val_logs["target_img1"].ndim == 2:
            val_logs["target_img1"] = val_logs["target_img1"].unsqueeze(0)
        elif val_logs["target_img1"].ndim != 3:
            raise RuntimeError("Unexpected target_img1 size from validation_step.")
        if val_logs["target_img2"].ndim == 2:
            val_logs["target_img2"] = val_logs["target_img2"].unsqueeze(0)
        elif val_logs["target_img2"].ndim != 3:
            raise RuntimeError("Unexpected target_img2 size from validation_step.")
        if val_logs["target_img3"].ndim == 2:
            val_logs["target_img3"] = val_logs["target_img3"].unsqueeze(0)
        elif val_logs["target_img3"].ndim != 3:
            raise RuntimeError("Unexpected target_img3 size from validation_step.")
        if val_logs["target_img4"].ndim == 2:
            val_logs["target_img4"] = val_logs["target_img4"].unsqueeze(0)
        elif val_logs["target_img4"].ndim != 3:
            raise RuntimeError("Unexpected target_img4 size from validation_step.")
        if val_logs["target_img5"].ndim == 2:
            val_logs["target_img5"] = val_logs["target_img5"].unsqueeze(0)
        elif val_logs["target_img5"].ndim != 3:
            raise RuntimeError("Unexpected target_img5 size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target_t1 = val_logs["target_t1"][i].unsqueeze(0)
                target_t2 = val_logs["target_t2"][i].unsqueeze(0)
                target_pd = val_logs["target_pd"][i].unsqueeze(0)
                target_img1 = val_logs["target_img1"][i].unsqueeze(0)
                target_img2 = val_logs["target_img2"][i].unsqueeze(0)
                target_img3 = val_logs["target_img3"][i].unsqueeze(0)
                target_img4 = val_logs["target_img4"][i].unsqueeze(0)
                target_img5 = val_logs["target_img5"][i].unsqueeze(0)
                output_t1 = val_logs["output_t1"][i].unsqueeze(0)
                output_t2 = val_logs["output_t2"][i].unsqueeze(0)
                output_pd = val_logs["output_pd"][i].unsqueeze(0)
                output_ie = val_logs["output_ie"][i].unsqueeze(0)
                output_b1 = val_logs["output_b1"][i].unsqueeze(0)
                output_img1 = val_logs["output_img1"][i].unsqueeze(0)
                output_img2 = val_logs["output_img2"][i].unsqueeze(0)
                output_img3 = val_logs["output_img3"][i].unsqueeze(0)
                output_img4 = val_logs["output_img4"][i].unsqueeze(0)
                output_img5 = val_logs["output_img5"][i].unsqueeze(0)

                target_pd = target_pd / target_pd.max()
                output_pd = output_pd / output_pd.max()
                error_t1 = torch.abs(target_t1 - output_t1)
                error_t2 = torch.abs(target_t2 - output_t2)
                error_pd = torch.abs(target_pd - output_pd)
                error_img1 = torch.abs(target_img1 - output_img1)
                error_img2 = torch.abs(target_img2 - output_img2)
                error_img3 = torch.abs(target_img3 - output_img3)
                error_img4 = torch.abs(target_img4 - output_img4)
                error_img5 = torch.abs(target_img5 - output_img5)

                output_t1 = output_t1 / output_t1.max()
                output_t2 = output_t2 / output_t2.max()
                output_pd = output_pd / output_pd.max()
                output_ie = output_ie / output_ie.max()
                output_b1 = output_b1 / output_b1.max()
                output_img1 = output_img1 / output_img1.max()
                output_img2 = output_img2 / output_img2.max()
                output_img3 = output_img3 / output_img3.max()
                output_img4 = output_img4 / output_img4.max()
                output_img5 = output_img5 / output_img5.max()
                target_t1 = target_t1 / target_t1.max()
                target_t2 = target_t2 / target_t2.max()
                target_pd = target_pd / target_pd.max()
                target_img1 = target_img1 / target_img1.max()
                target_img2 = target_img2 / target_img2.max()
                target_img3 = target_img3 / target_img3.max()
                target_img4 = target_img4 / target_img4.max()
                target_img5 = target_img5 / target_img5.max()
                error_t1 = error_t1 / error_t1.max()
                error_t2 = error_t2 / error_t2.max()
                error_pd = error_pd / error_pd.max()
                error_img1 = error_img1 / error_img1.max()
                error_img2 = error_img2 / error_img2.max()
                error_img3 = error_img3 / error_img3.max()
                error_img4 = error_img4 / error_img4.max()
                error_img5 = error_img5 / error_img5.max()

                self.log_image(f"{key}/target", torch.cat((target_t1,target_t2,target_pd),-1))
                self.log_image(f"{key}/target_forward", torch.cat((target_img1,target_img2,target_img3,target_img4,target_img5),-1))
                self.log_image(f"{key}/reconstruction", torch.cat((output_t1,output_t2,output_pd,output_ie,output_b1),-1))
                self.log_image(f"{key}/reconstruction_init_forward", torch.cat((output_img1,output_img2,output_img3,output_img4,output_img5),-1))
                self.log_image(f"{key}/error", torch.cat((error_t1,error_t2,error_pd),-1))
                self.log_image(f"{key}/error_init_forward", torch.cat((error_img1,error_img2,error_img3,error_img4,error_img5),-1))

        # compute evaluation metrics
        mse_vals_t1 = defaultdict(dict)
        mse_vals_t2 = defaultdict(dict)
        mse_vals_pd = defaultdict(dict)
        mse_vals_img1 = defaultdict(dict)
        mse_vals_img2 = defaultdict(dict)
        mse_vals_img3 = defaultdict(dict)
        mse_vals_img4 = defaultdict(dict)
        mse_vals_img5 = defaultdict(dict)
        target_norms_t1 = defaultdict(dict)
        target_norms_t2 = defaultdict(dict)
        target_norms_pd = defaultdict(dict)
        target_norms_img1 = defaultdict(dict)
        target_norms_img2 = defaultdict(dict)
        target_norms_img3 = defaultdict(dict)
        target_norms_img4 = defaultdict(dict)
        target_norms_img5 = defaultdict(dict)
        ssim_vals_t1 = defaultdict(dict)
        ssim_vals_t2 = defaultdict(dict)
        ssim_vals_pd = defaultdict(dict)
        ssim_vals_img1 = defaultdict(dict)
        ssim_vals_img2 = defaultdict(dict)
        ssim_vals_img3 = defaultdict(dict)
        ssim_vals_img4 = defaultdict(dict)
        ssim_vals_img5 = defaultdict(dict)
        max_vals_t1 = dict()
        max_vals_t2 = dict()
        max_vals_pd = dict()
        max_vals_img1 = dict()
        max_vals_img2 = dict()
        max_vals_img3 = dict()
        max_vals_img4 = dict()
        max_vals_img5 = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval_t1 = val_logs["max_value_t1"][i].cpu().numpy()
            maxval_t2 = val_logs["max_value_t2"][i].cpu().numpy()
            maxval_pd = val_logs["max_value_pd"][i].cpu().numpy()
            output_t1 = val_logs["output_t1"][i].cpu().numpy()
            output_t2 = val_logs["output_t2"][i].cpu().numpy()
            output_pd = val_logs["output_pd"][i].cpu().numpy()
            output_pd = output_pd / output_pd.max()
            output_img1 = val_logs["output_img1"][i].cpu().numpy()
            output_img2 = val_logs["output_img2"][i].cpu().numpy()
            output_img3 = val_logs["output_img3"][i].cpu().numpy()
            output_img4 = val_logs["output_img4"][i].cpu().numpy()
            output_img5 = val_logs["output_img5"][i].cpu().numpy()
            target_t1 = val_logs["target_t1"][i].cpu().numpy()
            target_t2 = val_logs["target_t2"][i].cpu().numpy()
            target_pd = val_logs["target_pd"][i].cpu().numpy()
            target_pd = target_pd / target_pd.max()
            target_img1 = val_logs["target_img1"][i].cpu().numpy()
            target_img2 = val_logs["target_img2"][i].cpu().numpy()
            target_img3 = val_logs["target_img3"][i].cpu().numpy()
            target_img4 = val_logs["target_img4"][i].cpu().numpy()
            target_img5 = val_logs["target_img5"][i].cpu().numpy()

            mse_vals_t1[fname][slice_num] = torch.tensor(
                evaluate.mse(target_t1, output_t1)
            ).view(1)
            mse_vals_t2[fname][slice_num] = torch.tensor(
                evaluate.mse(target_t2, output_t2)
            ).view(1)
            mse_vals_pd[fname][slice_num] = torch.tensor(
                evaluate.mse(target_pd, output_pd)
            ).view(1)
            mse_vals_img1[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img1, output_img1)
            ).view(1)
            mse_vals_img2[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img2, output_img2)
            ).view(1)
            mse_vals_img3[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img3, output_img3)
            ).view(1)
            mse_vals_img4[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img4, output_img4)
            ).view(1)
            mse_vals_img5[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img5, output_img5)
            ).view(1)
            target_norms_t1[fname][slice_num] = torch.tensor(
                evaluate.mse(target_t1, np.zeros_like(target_t1))
            ).view(1)
            target_norms_t2[fname][slice_num] = torch.tensor(
                evaluate.mse(target_t2, np.zeros_like(target_t2))
            ).view(1)
            target_norms_pd[fname][slice_num] = torch.tensor(
                evaluate.mse(target_pd, np.zeros_like(target_pd))
            ).view(1)
            target_norms_img1[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img1, np.zeros_like(target_img1))
            ).view(1)
            target_norms_img2[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img2, np.zeros_like(target_img2))
            ).view(1)
            target_norms_img3[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img3, np.zeros_like(target_img3))
            ).view(1)
            target_norms_img4[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img4, np.zeros_like(target_img4))
            ).view(1)
            target_norms_img5[fname][slice_num] = torch.tensor(
                evaluate.mse(target_img5, np.zeros_like(target_img5))
            ).view(1)
            ssim_vals_t1[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_t1[None, ...], output_t1[None, ...], maxval=maxval_t1)
            ).view(1)
            ssim_vals_t2[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_t2[None, ...], output_t2[None, ...], maxval=maxval_t2)
            ).view(1)
            ssim_vals_pd[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_pd[None, ...], output_pd[None, ...], maxval=maxval_pd)
            ).view(1)
            ssim_vals_img1[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_img1[None, ...], output_img1[None, ...], maxval=target_img1.max())
            ).view(1)
            ssim_vals_img2[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_img2[None, ...], output_img2[None, ...], maxval=target_img2.max())
            ).view(1)
            ssim_vals_img3[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_img3[None, ...], output_img3[None, ...], maxval=target_img3.max())
            ).view(1)
            ssim_vals_img4[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_img4[None, ...], output_img4[None, ...], maxval=target_img4.max())
            ).view(1)
            ssim_vals_img5[fname][slice_num] = torch.tensor(
                evaluate.ssim(target_img5[None, ...], output_img5[None, ...], maxval=target_img5.max())
            ).view(1)
            max_vals_t1[fname] = maxval_t1
            max_vals_t2[fname] = maxval_t2
            max_vals_pd[fname] = maxval_pd
            max_vals_img1[fname] = np.ones_like(maxval_t1)*target_img1.max()
            max_vals_img2[fname] = np.ones_like(maxval_t1)*target_img2.max()
            max_vals_img3[fname] = np.ones_like(maxval_t1)*target_img3.max()
            max_vals_img4[fname] = np.ones_like(maxval_t1)*target_img4.max()
            max_vals_img5[fname] = np.ones_like(maxval_t1)*target_img5.max()

        return {
            "val_loss_t1": val_logs["val_loss_t1"],
            "val_loss_t2": val_logs["val_loss_t2"],
            "val_loss_pd": val_logs["val_loss_pd"],
            "val_loss_img1": val_logs["val_loss_img1"],
            "val_loss_img2": val_logs["val_loss_img2"],
            "val_loss_img3": val_logs["val_loss_img3"],
            "val_loss_img4": val_logs["val_loss_img4"],
            "val_loss_img5": val_logs["val_loss_img5"],
            # "val_loss_img1_tv": val_logs["val_loss_img1_tv"],
            # "val_loss_img2_tv": val_logs["val_loss_img2_tv"],
            # "val_loss_img3_tv": val_logs["val_loss_img3_tv"],
            # "val_loss_img4_tv": val_logs["val_loss_img4_tv"],
            # "val_loss_img5_tv": val_logs["val_loss_img5_tv"],
            "mse_vals_t1": dict(mse_vals_t1),
            "mse_vals_t2": dict(mse_vals_t2),
            "mse_vals_pd": dict(mse_vals_pd),
            "mse_vals_img1": dict(mse_vals_img1),
            "mse_vals_img2": dict(mse_vals_img2),
            "mse_vals_img3": dict(mse_vals_img3),
            "mse_vals_img4": dict(mse_vals_img4),
            "mse_vals_img5": dict(mse_vals_img5),
            "target_norms_t1": dict(target_norms_t1),
            "target_norms_t2": dict(target_norms_t2),
            "target_norms_pd": dict(target_norms_pd),
            "target_norms_img1": dict(target_norms_img1),
            "target_norms_img2": dict(target_norms_img2),
            "target_norms_img3": dict(target_norms_img3),
            "target_norms_img4": dict(target_norms_img4),
            "target_norms_img5": dict(target_norms_img5),
            "ssim_vals_t1": dict(ssim_vals_t1),
            "ssim_vals_t2": dict(ssim_vals_t2),
            "ssim_vals_pd": dict(ssim_vals_pd),
            "ssim_vals_img1": dict(ssim_vals_img1),
            "ssim_vals_img2": dict(ssim_vals_img2),
            "ssim_vals_img3": dict(ssim_vals_img3),
            "ssim_vals_img4": dict(ssim_vals_img4),
            "ssim_vals_img5": dict(ssim_vals_img5),
            "max_vals_t1": max_vals_t1,
            "max_vals_t2": max_vals_t2,
            "max_vals_pd": max_vals_pd,
            "max_vals_img1": max_vals_img1,
            "max_vals_img2": max_vals_img2,
            "max_vals_img3": max_vals_img3,
            "max_vals_img4": max_vals_img4,
            "max_vals_img5": max_vals_img5,
            "loss_weight_t1": val_logs["loss_weight_t1"],
            "loss_weight_t2": val_logs["loss_weight_t2"],
            "loss_weight_pd": val_logs["loss_weight_pd"],
            "loss_weight_img1": val_logs["loss_weight_img1"],
            "loss_weight_img2": val_logs["loss_weight_img2"],
            "loss_weight_img3": val_logs["loss_weight_img3"],
            "loss_weight_img4": val_logs["loss_weight_img4"],
            "loss_weight_img5": val_logs["loss_weight_img5"],
            # "loss_weight_img1_tv": val_logs["loss_weight_img1_tv"],
            # "loss_weight_img2_tv": val_logs["loss_weight_img2_tv"],
            # "loss_weight_img3_tv": val_logs["loss_weight_img3_tv"],
            # "loss_weight_img4_tv": val_logs["loss_weight_img4_tv"],
            # "loss_weight_img5_tv": val_logs["loss_weight_img5_tv"],
        }

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses_t1 = []
        losses_t2 = []
        losses_pd = []
        losses_img1 = []
        losses_img2 = []
        losses_img3 = []
        losses_img4 = []
        losses_img5 = []
        # losses_img1_tv = []
        # losses_img2_tv = []
        # losses_img3_tv = []
        # losses_img4_tv = []
        # losses_img5_tv = []
        mse_vals_t1 = defaultdict(dict)
        mse_vals_t2 = defaultdict(dict)
        mse_vals_pd = defaultdict(dict)
        mse_vals_img1 = defaultdict(dict)
        mse_vals_img2 = defaultdict(dict)
        mse_vals_img3 = defaultdict(dict)
        mse_vals_img4 = defaultdict(dict)
        mse_vals_img5 = defaultdict(dict)
        target_norms_t1 = defaultdict(dict)
        target_norms_t2 = defaultdict(dict)
        target_norms_pd = defaultdict(dict)
        target_norms_img1 = defaultdict(dict)
        target_norms_img2 = defaultdict(dict)
        target_norms_img3 = defaultdict(dict)
        target_norms_img4 = defaultdict(dict)
        target_norms_img5 = defaultdict(dict)
        ssim_vals_t1 = defaultdict(dict)
        ssim_vals_t2 = defaultdict(dict)
        ssim_vals_pd = defaultdict(dict)
        ssim_vals_img1 = defaultdict(dict)
        ssim_vals_img2 = defaultdict(dict)
        ssim_vals_img3 = defaultdict(dict)
        ssim_vals_img4 = defaultdict(dict)
        ssim_vals_img5 = defaultdict(dict)
        max_vals_t1 = dict()
        max_vals_t2 = dict()
        max_vals_pd = dict()
        max_vals_img1 = dict()
        max_vals_img2 = dict()
        max_vals_img3 = dict()
        max_vals_img4 = dict()
        max_vals_img5 = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses_t1.append(val_log["val_loss_t1"].view(-1))
            losses_t2.append(val_log["val_loss_t2"].view(-1))
            losses_pd.append(val_log["val_loss_pd"].view(-1))
            losses_img1.append(val_log["val_loss_img1"].view(-1))
            losses_img2.append(val_log["val_loss_img2"].view(-1))
            losses_img3.append(val_log["val_loss_img3"].view(-1))
            losses_img4.append(val_log["val_loss_img4"].view(-1))
            losses_img5.append(val_log["val_loss_img5"].view(-1))
            # losses_img1_tv.append(val_log["val_loss_img1_tv"].view(-1))
            # losses_img2_tv.append(val_log["val_loss_img2_tv"].view(-1))
            # losses_img3_tv.append(val_log["val_loss_img3_tv"].view(-1))
            # losses_img4_tv.append(val_log["val_loss_img4_tv"].view(-1))
            # losses_img5_tv.append(val_log["val_loss_img5_tv"].view(-1))
            

            for k in val_log["mse_vals_t1"].keys():
                mse_vals_t1[k].update(val_log["mse_vals_t1"][k])
            for k in val_log["mse_vals_t2"].keys():
                mse_vals_t2[k].update(val_log["mse_vals_t2"][k])
            for k in val_log["mse_vals_pd"].keys():
                mse_vals_pd[k].update(val_log["mse_vals_pd"][k])
            for k in val_log["mse_vals_img1"].keys():
                mse_vals_img1[k].update(val_log["mse_vals_img1"][k])
            for k in val_log["mse_vals_img2"].keys():
                mse_vals_img2[k].update(val_log["mse_vals_img2"][k])
            for k in val_log["mse_vals_img3"].keys():
                mse_vals_img3[k].update(val_log["mse_vals_img3"][k])
            for k in val_log["mse_vals_img4"].keys():
                mse_vals_img4[k].update(val_log["mse_vals_img4"][k])
            for k in val_log["mse_vals_img5"].keys():
                mse_vals_img5[k].update(val_log["mse_vals_img5"][k])
            for k in val_log["target_norms_t1"].keys():
                target_norms_t1[k].update(val_log["target_norms_t1"][k])
            for k in val_log["target_norms_t2"].keys():
                target_norms_t2[k].update(val_log["target_norms_t2"][k])
            for k in val_log["target_norms_pd"].keys():
                target_norms_pd[k].update(val_log["target_norms_pd"][k])
            for k in val_log["target_norms_img1"].keys():
                target_norms_img1[k].update(val_log["target_norms_img1"][k])
            for k in val_log["target_norms_img2"].keys():
                target_norms_img2[k].update(val_log["target_norms_img2"][k])
            for k in val_log["target_norms_img3"].keys():
                target_norms_img3[k].update(val_log["target_norms_img3"][k])
            for k in val_log["target_norms_img4"].keys():
                target_norms_img4[k].update(val_log["target_norms_img4"][k])
            for k in val_log["target_norms_img5"].keys():
                target_norms_img5[k].update(val_log["target_norms_img5"][k])
            for k in val_log["ssim_vals_t1"].keys():
                ssim_vals_t1[k].update(val_log["ssim_vals_t1"][k])
            for k in val_log["ssim_vals_t2"].keys():
                ssim_vals_t2[k].update(val_log["ssim_vals_t2"][k])
            for k in val_log["ssim_vals_pd"].keys():
                ssim_vals_pd[k].update(val_log["ssim_vals_pd"][k])
            for k in val_log["ssim_vals_img1"].keys():
                ssim_vals_img1[k].update(val_log["ssim_vals_img1"][k])
            for k in val_log["ssim_vals_img2"].keys():
                ssim_vals_img2[k].update(val_log["ssim_vals_img2"][k])
            for k in val_log["ssim_vals_img3"].keys():
                ssim_vals_img3[k].update(val_log["ssim_vals_img3"][k])
            for k in val_log["ssim_vals_img4"].keys():
                ssim_vals_img4[k].update(val_log["ssim_vals_img4"][k])
            for k in val_log["ssim_vals_img5"].keys():
                ssim_vals_img5[k].update(val_log["ssim_vals_img5"][k])
            for k in val_log["max_vals_t1"]:
                max_vals_t1[k] = val_log["max_vals_t1"][k]
            for k in val_log["max_vals_t2"]:
                max_vals_t2[k] = val_log["max_vals_t2"][k]
            for k in val_log["max_vals_pd"]:
                max_vals_pd[k] = val_log["max_vals_pd"][k]
            for k in val_log["max_vals_img1"]:
                max_vals_img1[k] = val_log["max_vals_img1"][k]
            for k in val_log["max_vals_img2"]:
                max_vals_img2[k] = val_log["max_vals_img2"][k]
            for k in val_log["max_vals_img3"]:
                max_vals_img3[k] = val_log["max_vals_img3"][k]
            for k in val_log["max_vals_img4"]:
                max_vals_img4[k] = val_log["max_vals_img4"][k]
            for k in val_log["max_vals_img5"]:
                max_vals_img5[k] = val_log["max_vals_img5"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals_t1.keys()
            == mse_vals_t2.keys()
            == mse_vals_pd.keys()
            == mse_vals_img1.keys()
            == mse_vals_img2.keys()
            == mse_vals_img3.keys()
            == mse_vals_img4.keys()
            == mse_vals_img5.keys()
            == target_norms_t1.keys()
            == target_norms_t2.keys()
            == target_norms_pd.keys()
            == target_norms_img1.keys()
            == target_norms_img2.keys()
            == target_norms_img3.keys()
            == target_norms_img4.keys()
            == target_norms_img5.keys()
            == ssim_vals_t1.keys()
            == ssim_vals_t2.keys()
            == ssim_vals_pd.keys()
            == ssim_vals_img1.keys()
            == ssim_vals_img2.keys()
            == ssim_vals_img3.keys()
            == ssim_vals_img4.keys()
            == ssim_vals_img5.keys()
            == max_vals_t1.keys()
            == max_vals_t2.keys()
            == max_vals_pd.keys()
            == max_vals_img1.keys()
            == max_vals_img2.keys()
            == max_vals_img3.keys()
            == max_vals_img4.keys()
            == max_vals_img5.keys()
        )

        # apply means across image volumes
        metrics = {"nmse_t1": 0, "nmse_t2": 0, "nmse_pd": 0, \
                    "nmse_img1": 0, "nmse_img2": 0, "nmse_img3": 0, "nmse_img4": 0, "nmse_img5": 0, \
                    "ssim_t1": 0, "ssim_t2": 0, "ssim_pd": 0, \
                    "ssim_img1": 0, "ssim_img2": 0, "ssim_img3": 0, "ssim_img4": 0, "ssim_img5": 0, \
                    "psnr_t1": 0, "psnr_t2": 0, "psnr_pd": 0, \
                    "psnr_img1": 0, "psnr_img2": 0, "psnr_img3": 0, "psnr_img4": 0, "psnr_img5": 0}
        local_examples = 0
        for fname in mse_vals_t1.keys():
            local_examples = local_examples + 1
            mse_val_t1 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_t1[fname].items()])
            )
            mse_val_t2 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_t2[fname].items()])
            )
            mse_val_pd = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_pd[fname].items()])
            )
            mse_val_img1 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_img1[fname].items()])
            )
            mse_val_img2 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_img2[fname].items()])
            )
            mse_val_img3 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_img3[fname].items()])
            )
            mse_val_img4 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_img4[fname].items()])
            )
            mse_val_img5 = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals_img5[fname].items()])
            )
            target_norm_t1 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_t1[fname].items()])
            )
            target_norm_t2 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_t2[fname].items()])
            )
            target_norm_pd = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_pd[fname].items()])
            )
            target_norm_img1 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_img1[fname].items()])
            )
            target_norm_img2 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_img2[fname].items()])
            )
            target_norm_img3 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_img3[fname].items()])
            )
            target_norm_img4 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_img4[fname].items()])
            )
            target_norm_img5 = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms_img5[fname].items()])
            )
            metrics["nmse_t1"] = metrics["nmse_t1"] + mse_val_t1 / target_norm_t1
            metrics["nmse_t2"] = metrics["nmse_t2"] + mse_val_t2 / target_norm_t2
            metrics["nmse_pd"] = metrics["nmse_pd"] + mse_val_pd / target_norm_pd
            metrics["nmse_img1"] = metrics["nmse_img1"] + mse_val_img1 / target_norm_img1
            metrics["nmse_img2"] = metrics["nmse_img2"] + mse_val_img2 / target_norm_img2
            metrics["nmse_img3"] = metrics["nmse_img3"] + mse_val_img3 / target_norm_img3
            metrics["nmse_img4"] = metrics["nmse_img4"] + mse_val_img4 / target_norm_img4
            metrics["nmse_img5"] = metrics["nmse_img5"] + mse_val_img5 / target_norm_img5
            metrics["psnr_t1"] = (
                metrics["psnr_t1"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_t1[fname], dtype=mse_val_t1.dtype, device=mse_val_t1.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_t1)
            )
            metrics["psnr_t2"] = (
                metrics["psnr_t2"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_t2[fname], dtype=mse_val_t2.dtype, device=mse_val_t2.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_t2)
            )
            metrics["psnr_pd"] = (
                metrics["psnr_pd"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_pd[fname], dtype=mse_val_pd.dtype, device=mse_val_pd.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_pd)
            )
            metrics["psnr_img1"] = (
                metrics["psnr_img1"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_img1[fname], dtype=mse_val_img1.dtype, device=mse_val_img1.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_img1)
            )
            metrics["psnr_img2"] = (
                metrics["psnr_img2"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_img2[fname], dtype=mse_val_img2.dtype, device=mse_val_img2.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_img2)
            )
            metrics["psnr_img3"] = (
                metrics["psnr_img3"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_img3[fname], dtype=mse_val_img3.dtype, device=mse_val_img3.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_img3)
            )
            metrics["psnr_img4"] = (
                metrics["psnr_img4"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_img4[fname], dtype=mse_val_img4.dtype, device=mse_val_img4.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_img4)
            )
            metrics["psnr_img5"] = (
                metrics["psnr_img5"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals_img5[fname], dtype=mse_val_img5.dtype, device=mse_val_img5.device
                    )[0]
                )
                - 10 * torch.log10(mse_val_img5)
            )
            metrics["ssim_t1"] = metrics["ssim_t1"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_t1[fname].items()])
            )
            metrics["ssim_t2"] = metrics["ssim_t2"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_t2[fname].items()])
            )
            metrics["ssim_pd"] = metrics["ssim_pd"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_pd[fname].items()])
            )
            metrics["ssim_img1"] = metrics["ssim_img1"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_img1[fname].items()])
            )
            metrics["ssim_img2"] = metrics["ssim_img2"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_img2[fname].items()])
            )
            metrics["ssim_img3"] = metrics["ssim_img3"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_img3[fname].items()])
            )
            metrics["ssim_img4"] = metrics["ssim_img4"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_img4[fname].items()])
            )
            metrics["ssim_img5"] = metrics["ssim_img5"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals_img5[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse_t1"] = self.NMSE(metrics["nmse_t1"])
        metrics["nmse_t2"] = self.NMSE(metrics["nmse_t2"])
        metrics["nmse_pd"] = self.NMSE(metrics["nmse_pd"])
        metrics["nmse_img1"] = self.NMSE(metrics["nmse_img1"])
        metrics["nmse_img2"] = self.NMSE(metrics["nmse_img2"])
        metrics["nmse_img3"] = self.NMSE(metrics["nmse_img3"])
        metrics["nmse_img4"] = self.NMSE(metrics["nmse_img4"])
        metrics["nmse_img5"] = self.NMSE(metrics["nmse_img5"])
        metrics["ssim_t1"] = self.SSIM(metrics["ssim_t1"])
        metrics["ssim_t2"] = self.SSIM(metrics["ssim_t2"])
        metrics["ssim_pd"] = self.SSIM(metrics["ssim_pd"])
        metrics["ssim_img1"] = self.SSIM(metrics["ssim_img1"])
        metrics["ssim_img2"] = self.SSIM(metrics["ssim_img2"])
        metrics["ssim_img3"] = self.SSIM(metrics["ssim_img3"])
        metrics["ssim_img4"] = self.SSIM(metrics["ssim_img4"])
        metrics["ssim_img5"] = self.SSIM(metrics["ssim_img5"])
        metrics["psnr_t1"] = self.PSNR(metrics["psnr_t1"])
        metrics["psnr_t2"] = self.PSNR(metrics["psnr_t2"])
        metrics["psnr_pd"] = self.PSNR(metrics["psnr_pd"])
        metrics["psnr_img1"] = self.PSNR(metrics["psnr_img1"])
        metrics["psnr_img2"] = self.PSNR(metrics["psnr_img2"])
        metrics["psnr_img3"] = self.PSNR(metrics["psnr_img3"])
        metrics["psnr_img4"] = self.PSNR(metrics["psnr_img4"])
        metrics["psnr_img5"] = self.PSNR(metrics["psnr_img5"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss((torch.sum(torch.cat(losses_t1)) * val_log["loss_weight_t1"] + torch.sum(torch.cat(losses_t2)) * val_log["loss_weight_t2"] + \
                                    torch.sum(torch.cat(losses_pd)) * val_log["loss_weight_pd"] + \
                                    torch.sum(torch.cat(losses_img1)) * val_log["loss_weight_img1"] + torch.sum(torch.cat(losses_img2)) * val_log["loss_weight_img2"] + \
                                    torch.sum(torch.cat(losses_img3)) * val_log["loss_weight_img3"] + torch.sum(torch.cat(losses_img4)) * val_log["loss_weight_img4"] + \
                                    torch.sum(torch.cat(losses_img5)) * val_log["loss_weight_img5"]) \
                                / (val_log["loss_weight_t1"] + val_log["loss_weight_t2"] + val_log["loss_weight_pd"] + \
                                    val_log["loss_weight_img1"] + val_log["loss_weight_img2"] + val_log["loss_weight_img3"] + val_log["loss_weight_img4"] + val_log["loss_weight_img5"]))
        # val_loss = self.ValLoss((torch.sum(torch.cat(losses_t1)) * val_log["loss_weight_t1"] + torch.sum(torch.cat(losses_t2)) * val_log["loss_weight_t2"] + \
        #                             torch.sum(torch.cat(losses_pd)) * val_log["loss_weight_pd"] + \
        #                             torch.sum(torch.cat(losses_img1)) * val_log["loss_weight_img1"] + torch.sum(torch.cat(losses_img2)) * val_log["loss_weight_img2"] + \
        #                             torch.sum(torch.cat(losses_img3)) * val_log["loss_weight_img3"] + torch.sum(torch.cat(losses_img4)) * val_log["loss_weight_img4"] + \
        #                             torch.sum(torch.cat(losses_img5)) * val_log["loss_weight_img5"] + \
        #                             torch.sum(torch.cat(losses_img1_tv)) * val_log["loss_weight_img1_tv"] + torch.sum(torch.cat(losses_img2_tv)) * val_log["loss_weight_img2_tv"] + \
        #                             torch.sum(torch.cat(losses_img3_tv)) * val_log["loss_weight_img3_tv"] + torch.sum(torch.cat(losses_img4_tv)) * val_log["loss_weight_img4_tv"] + \
        #                             torch.sum(torch.cat(losses_img5_tv)) * val_log["loss_weight_img5_tv"]) \
        #                         / (val_log["loss_weight_t1"] + val_log["loss_weight_t2"] + val_log["loss_weight_pd"] + \
        #                             val_log["loss_weight_img1"] + val_log["loss_weight_img2"] + val_log["loss_weight_img3"] + val_log["loss_weight_img4"] + val_log["loss_weight_img5"] + \
        #                             val_log["loss_weight_img1_tv"] + val_log["loss_weight_img2_tv"] + val_log["loss_weight_img3_tv"] + val_log["loss_weight_img4_tv"] + val_log["loss_weight_img5_tv"]))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses_t1), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

    def test_epoch_end(self, test_logs):
        outputs_t1 = defaultdict(dict)
        outputs_t2 = defaultdict(dict)
        outputs_pd = defaultdict(dict)
        outputs_ie = defaultdict(dict)
        outputs_b1 = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs_t1[fname][int(slice_num.cpu())] = log["output_t1"][i]
                outputs_t2[fname][int(slice_num.cpu())] = log["output_t2"][i]
                outputs_pd[fname][int(slice_num.cpu())] = log["output_pd"][i]
                outputs_ie[fname][int(slice_num.cpu())] = log["output_ie"][i]
                outputs_b1[fname][int(slice_num.cpu())] = log["output_b1"][i]

        # stack all the slices for each file
        for fname in outputs_t1:
            outputs_t1[fname] = np.stack(
                [out for _, out in sorted(outputs_t1[fname].items())]
            )
            outputs_t2[fname] = np.stack(
                [out for _, out in sorted(outputs_t2[fname].items())]
            )
            outputs_pd[fname] = np.stack(
                [out for _, out in sorted(outputs_pd[fname].items())]
            )
            outputs_ie[fname] = np.stack(
                [out for _, out in sorted(outputs_ie[fname].items())]
            )
            outputs_b1[fname] = np.stack(
                [out for _, out in sorted(outputs_b1[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        fastmri.save_reconstructions_qalas(outputs_t1, outputs_t2, outputs_pd, outputs_ie, outputs_b1, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser
