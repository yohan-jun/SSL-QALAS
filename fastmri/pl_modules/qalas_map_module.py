"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms_qalas
from fastmri.models import QALAS_MAP

from .mri_module_qalas_map import MriModuleQALAS_MAP
import numpy as np


class QALAS_MAPModule(MriModuleQALAS_MAP):
    """
    QALAS training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        maps_chans: int = 32,
        maps_layers: int = 5,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.maps_chans = maps_chans
        self.maps_layers = maps_layers
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.qalas = QALAS_MAP(
            num_cascades=self.num_cascades,
            maps_chans=self.maps_chans,
            maps_layers=self.maps_layers,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss_l2_t1 = torch.nn.MSELoss()
        self.loss_l2_t2 = torch.nn.MSELoss()
        self.loss_l2_pd = torch.nn.MSELoss()
        self.loss_l2_img1 = torch.nn.MSELoss()
        self.loss_l2_img2 = torch.nn.MSELoss()
        self.loss_l2_img3 = torch.nn.MSELoss()
        self.loss_l2_img4 = torch.nn.MSELoss()
        self.loss_l2_img5 = torch.nn.MSELoss()

    def forward(self, masked_kspace_acq1, masked_kspace_acq2, masked_kspace_acq3, masked_kspace_acq4, masked_kspace_acq5, \
                mask_acq1, mask_acq2, mask_acq3, mask_acq4, mask_acq5, mask_brain, \
                # coil_sens, \
                b1, ie, max_value_t1, max_value_t2, max_value_pd, num_low_frequencies):
        return self.qalas(masked_kspace_acq1, masked_kspace_acq2, masked_kspace_acq3, masked_kspace_acq4, masked_kspace_acq5, \
                        mask_acq1, mask_acq2, mask_acq3, mask_acq4, mask_acq5, mask_brain, \
                        # coil_sens, \
                        b1, ie, max_value_t1, max_value_t2, max_value_pd, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        output_t1, output_t2, output_pd, output_ie, output_b1, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            self(batch.masked_kspace_acq1, batch.masked_kspace_acq2, batch.masked_kspace_acq3, batch.masked_kspace_acq4, batch.masked_kspace_acq5, \
                batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                # batch.coil_sens, \
                batch.b1, batch.ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies)

        # If raw k-space data were used, use following 5 lines
        # img_acq1 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq1), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq1.shape[2] * batch.masked_kspace_acq1.shape[3])
        # img_acq2 = -fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq2), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq2.shape[2] * batch.masked_kspace_acq2.shape[3])
        # img_acq3 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq3), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq3.shape[2] * batch.masked_kspace_acq3.shape[3])
        # img_acq4 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq4), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq4.shape[2] * batch.masked_kspace_acq4.shape[3])
        # img_acq5 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq5), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq5.shape[2] * batch.masked_kspace_acq5.shape[3])

        # If DICOM data were used, use following 5 lines
        img_acq1 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq1[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq1.shape[2] * batch.masked_kspace_acq1.shape[3])
        img_acq2 = -fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq2[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq2.shape[2] * batch.masked_kspace_acq2.shape[3])
        img_acq3 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq3[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq3.shape[2] * batch.masked_kspace_acq3.shape[3])
        img_acq4 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq4[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq4.shape[2] * batch.masked_kspace_acq4.shape[3])
        img_acq5 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq5[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq5.shape[2] * batch.masked_kspace_acq5.shape[3])

        target_t1, output_t1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_t1)
        target_t2, output_t2 = transforms_qalas.center_crop_to_smallest(batch.target_t2, output_t2)
        target_pd, output_pd = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_pd)
        target_pd, output_ie = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_ie)
        target_pd, output_b1 = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_b1)

        target_t1, img_acq1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq1.squeeze(1))
        target_t1, img_acq2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq2.squeeze(1))
        target_t1, img_acq3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq3.squeeze(1))
        target_t1, img_acq4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq4.squeeze(1))
        target_t1, img_acq5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq5.squeeze(1))

        target_t1, output_img1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img1.squeeze(1))
        target_t1, output_img2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img2.squeeze(1))
        target_t1, output_img3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img3.squeeze(1))
        target_t1, output_img4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img4.squeeze(1))
        target_t1, output_img5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img5.squeeze(1))

        target_t1 = target_t1 * batch.mask_brain
        target_t2 = target_t2 * batch.mask_brain
        target_pd = target_pd * batch.mask_brain

        output_t1 = output_t1 * batch.mask_brain
        output_t2 = output_t2 * batch.mask_brain
        output_pd = output_pd * batch.mask_brain
        output_ie = output_ie * batch.mask_brain
        output_b1 = output_b1 * batch.mask_brain

        img_acq1 = img_acq1 * batch.mask_brain
        img_acq2 = img_acq2 * batch.mask_brain
        img_acq3 = img_acq3 * batch.mask_brain
        img_acq4 = img_acq4 * batch.mask_brain
        img_acq5 = img_acq5 * batch.mask_brain

        output_img1 = output_img1 * batch.mask_brain
        output_img2 = output_img2 * batch.mask_brain
        output_img3 = output_img3 * batch.mask_brain
        output_img4 = output_img4 * batch.mask_brain
        output_img5 = output_img5 * batch.mask_brain

        if batch.mask_brain.sum() == 0:
            target_t1 = target_t1 + 1e-5
            target_t2 = target_t2 + 1e-5
            target_pd = target_pd + 1e-5
            output_t1 = output_t1 + 1e-5
            output_t2 = output_t2 + 1e-5
            output_pd = output_pd + 1e-5
            output_ie = output_ie + 1e-5
            output_b1 = output_b1 + 1e-5
            img_acq1 = img_acq1 + 1e-5
            img_acq2 = img_acq2 + 1e-5
            img_acq3 = img_acq3 + 1e-5
            img_acq4 = img_acq4 + 1e-5
            img_acq5 = img_acq5 + 1e-5
            output_img1 = output_img1 + 1e-5
            output_img2 = output_img2 + 1e-5
            output_img3 = output_img3 + 1e-5
            output_img4 = output_img4 + 1e-5
            output_img5 = output_img5 + 1e-5

        loss_t1 = self.loss_l2_t1(output_t1.unsqueeze(1), target_t1.unsqueeze(1))
        loss_t2 = self.loss_l2_t2(output_t2.unsqueeze(1), target_t2.unsqueeze(1))
        loss_pd = self.loss_l2_pd(output_pd.unsqueeze(1) / output_pd.max(), target_pd.unsqueeze(1) / target_pd.max())
        loss_img1 = self.loss_l2_img1(output_img1.unsqueeze(1), img_acq1.unsqueeze(1))
        loss_img2 = self.loss_l2_img2(output_img2.unsqueeze(1), img_acq2.unsqueeze(1))
        loss_img3 = self.loss_l2_img3(output_img3.unsqueeze(1), img_acq3.unsqueeze(1))
        loss_img4 = self.loss_l2_img4(output_img4.unsqueeze(1), img_acq4.unsqueeze(1))
        loss_img5 = self.loss_l2_img5(output_img5.unsqueeze(1), img_acq5.unsqueeze(1))
        def loss_tv(img):
            pixel_dif1 = img[..., 1:,:] - img[..., :-1,:]
            pixel_dif2 = img[..., :,1:] - img[..., :,:-1]
            reduce_axes = (-2,-1)
            res1 = pixel_dif1.abs().sum(dim=reduce_axes)
            res2 = pixel_dif2.abs().sum(dim=reduce_axes)
            return (res1 + res2) / (img.shape[-1] * img.shape[-2])
        loss_img1_tv = loss_tv(output_img1)
        loss_img2_tv = loss_tv(output_img2)
        loss_img3_tv = loss_tv(output_img3)
        loss_img4_tv = loss_tv(output_img4)
        loss_img5_tv = loss_tv(output_img5)

        loss_weight_t1 = 0
        loss_weight_t2 = 0
        loss_weight_pd = 0
        loss_weight_img1 = 1
        loss_weight_img2 = 1
        loss_weight_img3 = 1
        loss_weight_img4 = 1
        loss_weight_img5 = 1
        loss_weight_img1_tv = 0
        loss_weight_img2_tv = 0
        loss_weight_img3_tv = 0
        loss_weight_img4_tv = 0
        loss_weight_img5_tv = 0
        loss = (loss_t1 * loss_weight_t1 + loss_t2 * loss_weight_t2 + loss_pd * loss_weight_pd + \
                loss_img1 * loss_weight_img1 + loss_img2 * loss_weight_img2 + loss_img3 * loss_weight_img3 + loss_img4 * loss_weight_img4 + loss_img5 * loss_weight_img5) \
                 / (loss_weight_t1 + loss_weight_t2 + loss_weight_pd + \
                    loss_weight_img1 + loss_weight_img2 + loss_weight_img3 + loss_weight_img4 + loss_weight_img5)

        self.log("train_loss_t1", loss_t1)
        self.log("train_loss_t2", loss_t2)
        self.log("train_loss_pd", loss_pd)
        self.log("train_loss_img1", loss_img1)
        self.log("train_loss_img2", loss_img2)
        self.log("train_loss_img3", loss_img3)
        self.log("train_loss_img4", loss_img4)
        self.log("train_loss_img5", loss_img5)

        return loss


    def validation_step(self, batch, batch_idx):
        output_t1, output_t2, output_pd, output_ie, output_b1, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            self.forward(batch.masked_kspace_acq1, batch.masked_kspace_acq2, batch.masked_kspace_acq3, batch.masked_kspace_acq4, batch.masked_kspace_acq5, \
                        batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                        # batch.coil_sens, \
                        batch.b1, batch.ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies)

        # If raw k-space data were used, use following 5 lines
        # img_acq1 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq1), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq1.shape[2] * batch.masked_kspace_acq1.shape[3])
        # img_acq2 = -fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq2), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq2.shape[2] * batch.masked_kspace_acq2.shape[3])
        # img_acq3 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq3), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq3.shape[2] * batch.masked_kspace_acq3.shape[3])
        # img_acq4 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq4), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq4.shape[2] * batch.masked_kspace_acq4.shape[3])
        # img_acq5 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(batch.masked_kspace_acq5), fastmri.complex_conj(batch.coil_sens)).sum(dim=1, keepdim=True)) / np.sqrt(batch.masked_kspace_acq5.shape[2] * batch.masked_kspace_acq5.shape[3])

        # If DICOM data were used, use following 5 lines
        img_acq1 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq1[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq1.shape[2] * batch.masked_kspace_acq1.shape[3])
        img_acq2 = -fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq2[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq2.shape[2] * batch.masked_kspace_acq2.shape[3])
        img_acq3 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq3[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq3.shape[2] * batch.masked_kspace_acq3.shape[3])
        img_acq4 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq4[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq4.shape[2] * batch.masked_kspace_acq4.shape[3])
        img_acq5 = fastmri.complex_abs(fastmri.ifft2c(batch.masked_kspace_acq5[:,0:1,...])) / np.sqrt(batch.masked_kspace_acq5.shape[2] * batch.masked_kspace_acq5.shape[3])

        target_t1, output_t1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_t1)
        target_t2, output_t2 = transforms_qalas.center_crop_to_smallest(batch.target_t2, output_t2)
        target_pd, output_pd = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_pd)
        target_pd, output_ie = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_ie)
        target_pd, output_b1 = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_b1)

        target_t1, img_acq1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq1.squeeze(1))
        target_t1, img_acq2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq2.squeeze(1))
        target_t1, img_acq3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq3.squeeze(1))
        target_t1, img_acq4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq4.squeeze(1))
        target_t1, img_acq5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq5.squeeze(1))

        target_t1, output_img1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img1.squeeze(1))
        target_t1, output_img2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img2.squeeze(1))
        target_t1, output_img3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img3.squeeze(1))
        target_t1, output_img4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img4.squeeze(1))
        target_t1, output_img5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img5.squeeze(1))

        target_t1 = target_t1 * batch.mask_brain
        target_t2 = target_t2 * batch.mask_brain
        target_pd = target_pd * batch.mask_brain

        output_t1 = output_t1 * batch.mask_brain
        output_t2 = output_t2 * batch.mask_brain
        output_pd = output_pd * batch.mask_brain
        output_ie = output_ie * batch.mask_brain
        output_b1 = output_b1 * batch.mask_brain

        img_acq1 = img_acq1 * batch.mask_brain
        img_acq2 = img_acq2 * batch.mask_brain
        img_acq3 = img_acq3 * batch.mask_brain
        img_acq4 = img_acq4 * batch.mask_brain
        img_acq5 = img_acq5 * batch.mask_brain

        output_img1 = output_img1 * batch.mask_brain
        output_img2 = output_img2 * batch.mask_brain
        output_img3 = output_img3 * batch.mask_brain
        output_img4 = output_img4 * batch.mask_brain
        output_img5 = output_img5 * batch.mask_brain

        def loss_tv(img):
            pixel_dif1 = img[..., 1:,:] - img[..., :-1,:]
            pixel_dif2 = img[..., :,1:] - img[..., :,:-1]
            reduce_axes = (-2,-1)
            res1 = pixel_dif1.abs().sum(dim=reduce_axes)
            res2 = pixel_dif2.abs().sum(dim=reduce_axes)
            return (res1 + res2) / (img.shape[-1] * img.shape[-2])
        loss_img1_tv = loss_tv(output_img1)
        loss_img2_tv = loss_tv(output_img2)
        loss_img3_tv = loss_tv(output_img3)
        loss_img4_tv = loss_tv(output_img4)
        loss_img5_tv = loss_tv(output_img5)

        loss_weight_t1 = 0
        loss_weight_t2 = 0
        loss_weight_pd = 0
        loss_weight_img1 = 1
        loss_weight_img2 = 1
        loss_weight_img3 = 1
        loss_weight_img4 = 1
        loss_weight_img5 = 1
        loss_weight_img1_tv = 0
        loss_weight_img2_tv = 0
        loss_weight_img3_tv = 0
        loss_weight_img4_tv = 0
        loss_weight_img5_tv = 0

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value_t1": batch.max_value_t1,
            "max_value_t2": batch.max_value_t2,
            "max_value_pd": torch.ones_like(batch.max_value_t1),
            "output_t1": output_t1,
            "output_t2": output_t2,
            "output_pd": output_pd,
            "output_ie": output_ie,
            "output_b1": output_b1,
            "output_img1": torch.abs(output_img1),
            "output_img2": torch.abs(output_img2),
            "output_img3": torch.abs(output_img3),
            "output_img4": torch.abs(output_img4),
            "output_img5": torch.abs(output_img5),
            "target_t1": target_t1,
            "target_t2": target_t2,
            "target_pd": target_pd,
            "target_img1": torch.abs(img_acq1),
            "target_img2": torch.abs(img_acq2),
            "target_img3": torch.abs(img_acq3),
            "target_img4": torch.abs(img_acq4),
            "target_img5": torch.abs(img_acq5),
            "val_loss_t1": self.loss_l2_t1(output_t1.unsqueeze(1), target_t1.unsqueeze(1)),
            "val_loss_t2": self.loss_l2_t2(output_t2.unsqueeze(1), target_t2.unsqueeze(1)),
            "val_loss_pd": self.loss_l2_pd(output_pd.unsqueeze(1) / output_pd.max(), target_pd.unsqueeze(1) / target_pd.max()),
            "val_loss_img1": self.loss_l2_img1(output_img1.unsqueeze(1), img_acq1.unsqueeze(1)),
            "val_loss_img2": self.loss_l2_img2(output_img2.unsqueeze(1), img_acq2.unsqueeze(1)),
            "val_loss_img3": self.loss_l2_img3(output_img3.unsqueeze(1), img_acq3.unsqueeze(1)),
            "val_loss_img4": self.loss_l2_img4(output_img4.unsqueeze(1), img_acq4.unsqueeze(1)),
            "val_loss_img5": self.loss_l2_img5(output_img5.unsqueeze(1), img_acq5.unsqueeze(1)),
            "loss_weight_t1": loss_weight_t1,
            "loss_weight_t2": loss_weight_t2,
            "loss_weight_pd": loss_weight_pd,
            "loss_weight_img1": loss_weight_img1,
            "loss_weight_img2": loss_weight_img2,
            "loss_weight_img3": loss_weight_img3,
            "loss_weight_img4": loss_weight_img4,
            "loss_weight_img5": loss_weight_img5,
        }


    def test_step(self, batch, batch_idx):
        output_t1, output_t2, output_pd, output_ie, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            self(batch.masked_kspace_acq1, batch.masked_kspace_acq2, batch.masked_kspace_acq3, batch.masked_kspace_acq4, batch.masked_kspace_acq5, \
                batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                # batch.coil_sens, \
                batch.b1, batch.ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies)

        # check for FLAIR 203
        if output_t1.shape[-1] < batch.crop_size[1]:
            crop_size = (output_t1.shape[-1], output_t1.shape[-1])
        else:
            crop_size = batch.crop_size

        output_t1 = transforms_qalas.center_crop(output_t1, crop_size)
        output_t2 = transforms_qalas.center_crop(output_t2, crop_size)
        output_pd = transforms_qalas.center_crop(output_pd, crop_size)
        output_ie = transforms_qalas.center_crop(output_ie, crop_size)

        output_t1 = output_t1 * batch.mask_brain
        output_t2 = output_t2 * batch.mask_brain
        output_pd = output_pd * batch.mask_brain
        output_ie = output_ie * batch.mask_brain

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output_t1": output_t1.cpu().numpy(),
            "output_t2": output_t2.cpu().numpy(),
            "output_pd": output_pd.cpu().numpy(),
            "output_ie": output_ie.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModuleQALAS_MAP.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of QALAS cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in QALAS blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in QALAS blocks",
        )
        parser.add_argument(
            "--maps_chans",
            default=32,
            type=int,
            help="Number of channels for mapping CNN in QALAS",
        )
        parser.add_argument(
            "--maps_layers",
            default=5,
            type=float,
            help="Number of layers for mapping CNN in QALAS",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
