"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple, Optional

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms_qalas
import numpy as np

from .unet import Unet
from .cnn import CNN
eps = 1e-5

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 5,
        out_chans: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 1, c * h * w)

        mean = x.mean(dim=2).view(b, 1, 1, 1)
        std = x.std(dim=2).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if not x.shape[-1] == 2:
        #     raise ValueError("Last dimension must be 2 for complex.")

        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)

        return x


class MappingModel(nn.Module):
    """
    Mapping CNN model.
    """

    def __init__(
        self,
        chans: int,
        num_layers: int,
        num_pools: int = 3,
        in_chans: int = 5,
        out_chans: int = 4, # number of quantitative maps
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            in_chans: Number of channels in the input to the CNN model.
            out_chans: Number of channels in the output to the CNN model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.cnn = CNN(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_layers=num_layers,
            drop_prob=drop_prob,
        )
        # self.cnn = NormUnet(
        #     in_chans=in_chans,
        #     out_chans=out_chans,
        #     chans=chans,
        #     num_pools=num_pools,
        #     drop_prob=drop_prob,
        # )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.complex_to_chan_dim(x)
        x = self.cnn(x)

        return x


class QALAS_MAP(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 10,
        maps_chans: int = 64,
        maps_layers: int = 5,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.maps_net = MappingModel(
            chans=maps_chans,
            num_layers=maps_layers,
            drop_prob = 0.0,
        )
        self.cascades = nn.ModuleList(
            [QALASBlock() for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace_acq1: torch.Tensor,
        masked_kspace_acq2: torch.Tensor,
        masked_kspace_acq3: torch.Tensor,
        masked_kspace_acq4: torch.Tensor,
        masked_kspace_acq5: torch.Tensor,
        mask_acq1: torch.Tensor,
        mask_acq2: torch.Tensor,
        mask_acq3: torch.Tensor,
        mask_acq4: torch.Tensor,
        mask_acq5: torch.Tensor,
        mask_brain: torch.Tensor,
        # coil_sens: torch.Tensor,
        b1: torch.Tensor,
        ie: torch.Tensor,
        max_value_t1: torch.Tensor,
        max_value_t2: torch.Tensor,
        max_value_pd: torch.Tensor,
        max_value_ie: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        # If raw k-space data were used, use following 5 lines
        # image_pred_acq1 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(masked_kspace_acq1), fastmri.complex_conj(coil_sens)).sum(dim=1, keepdim=True) / np.sqrt(masked_kspace_acq1.shape[2] * masked_kspace_acq1.shape[3]))
        # image_pred_acq2 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(masked_kspace_acq2), fastmri.complex_conj(coil_sens)).sum(dim=1, keepdim=True) / np.sqrt(masked_kspace_acq2.shape[2] * masked_kspace_acq2.shape[3]))
        # image_pred_acq3 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(masked_kspace_acq3), fastmri.complex_conj(coil_sens)).sum(dim=1, keepdim=True) / np.sqrt(masked_kspace_acq3.shape[2] * masked_kspace_acq3.shape[3]))
        # image_pred_acq4 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(masked_kspace_acq4), fastmri.complex_conj(coil_sens)).sum(dim=1, keepdim=True) / np.sqrt(masked_kspace_acq4.shape[2] * masked_kspace_acq4.shape[3]))
        # image_pred_acq5 = fastmri.complex_abs(fastmri.complex_mul(fastmri.ifft2c(masked_kspace_acq5), fastmri.complex_conj(coil_sens)).sum(dim=1, keepdim=True) / np.sqrt(masked_kspace_acq5.shape[2] * masked_kspace_acq5.shape[3]))

        # If DICOM data were used, use following 5 lines
        image_pred_acq1 = masked_kspace_acq1
        image_pred_acq2 = masked_kspace_acq2
        image_pred_acq3 = masked_kspace_acq3
        image_pred_acq4 = masked_kspace_acq4
        image_pred_acq5 = masked_kspace_acq5

        # Using CNN for Mapping
        map_pred = self.maps_net(torch.cat((image_pred_acq1, image_pred_acq2, image_pred_acq3, image_pred_acq4, image_pred_acq5), 1))
        # map_pred = self.maps_net(torch.cat((image_pred_acq1, image_pred_acq2, image_pred_acq3, image_pred_acq4, image_pred_acq5, b1.unsqueeze(1).to(image_pred_acq1.device)), 1))

        map_pred_t1 = map_pred[:,0:1,:,:] * max_value_t1[0,:]
        map_pred_t2 = map_pred[:,1:2,:,:] * max_value_t2[0,:]
        # map_pred_pd = map_pred[:,2:3,:,:]
        map_pred_pd = map_pred[:,2:3,:,:] / torch.sin(np.pi / 180 * torch.Tensor([4]).to(map_pred.device))
        map_pred_ie = map_pred[:,3:4,:,:] * (1 - 0.5) + 0.5 # 0.5-1.0

        map_pred_b1 = b1.unsqueeze(1).to(map_pred.device)

        for cascade in self.cascades:
            [img_acq1, img_acq2, img_acq3, img_acq4, img_acq5] = cascade(map_pred_t1, map_pred_t2, map_pred_pd, map_pred_ie, map_pred_b1)
        return map_pred_t1.squeeze(1), map_pred_t2.squeeze(1), map_pred_pd.squeeze(1), map_pred_ie.squeeze(1), map_pred_b1.squeeze(1), \
                img_acq1, img_acq2, img_acq3, img_acq4, img_acq5


class QALASBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

    def qalas_forward_eq(self, x_t1: torch.Tensor, x_t2: torch.Tensor, x_m0: torch.Tensor, x_ie: torch.Tensor, x_b1: torch.Tensor) -> torch.Tensor:
        flip_ang = torch.Tensor([4]).to(x_t1.device) * x_b1.to(x_t1.device)     # Refocusing flip angle
        tf = torch.Tensor([128]).to(x_t1.device)                                # Turbo Factor
        esp = torch.Tensor([0.0058]).to(x_t1.device)                            # ESP
        etl = tf * esp                                                          #
        t2_prep = torch.Tensor([0.1097]).to(x_t1.device)                        # T2_prep
        gap_bw_ro = torch.Tensor([0.9]).to(x_t1.device)                         # Gap b/w readouts
        tr = torch.Tensor([4.5]).to(x_t1.device)                                # TR
        time_relax_end = torch.Tensor([0]).to(x_t1.device)                      # Relax time at the end
        echo2use = 1                                                            # Index of echo to use

        # Timings
        delt_m1_m2 = t2_prep                                # (t2_prep = 0.1097)
        delt_m0_m1 = gap_bw_ro - etl - delt_m1_m2           # (0.9 - 0.7296 - 0.1097 = 0.0607)
        delt_m2_m3 = etl                                    # Duration of readout #1 = 0.7296
        delt_m2_m6 = gap_bw_ro                              # Gap b/w readouts = 0.9
        delt_m4_m5 = torch.Tensor([0.0128]).to(x_t1.device)          # Inversion pulse = 0.0128
        delt_m5_m6 = torch.Tensor([0.1 - 0.00645]).to(x_t1.device)   # Gap b/w end of inversion pulse and start of readout #2 = 0.0355
        delt_m3_m4 = delt_m2_m6 - delt_m2_m3 - delt_m4_m5 - delt_m5_m6 # Gap b/w end of readout #1 and start of inversion pulse = 0.9 - 0.7296 - 0.0128 - 0.0355 = 0.1221
        delt_m6_m7 = etl                                    # Duration of readout #2 = 0.7296
        delt_m7_m8 = gap_bw_ro - etl                        # From end of readout #2 to begin of readout #3 = 0.1704
        delt_m8_m9 = etl                                    # Duration of readout #3 = 0.7296
        delt_m9_m10 = gap_bw_ro - etl                       # From end of readout #3 to begin of readout #4 = 0.1704
        delt_m10_m11 = etl                                  # Duration of readout #4 = 0.7296
        delt_m11_m12 = gap_bw_ro - etl                      # From end of readout #4 to begin of readout #5 = 0.1704
        delt_m12_m13 = etl                                  # Duration of readout #5 = 0.7296
        total_duration = delt_m0_m1 + delt_m1_m2 + delt_m2_m3 + delt_m3_m4 + delt_m4_m5 + delt_m5_m6 + delt_m6_m7 + \
                            delt_m7_m8 + delt_m8_m9 + delt_m9_m10 + delt_m10_m11 + delt_m11_m12 + delt_m12_m13
        delt_m13_end = torch.maximum(tr - total_duration, torch.Tensor([0]).to(x_t1.device))
        if time_relax_end > 0:
            delt_m13_end = delt_m13_end + time_relax_end

        # Const.
        ET2 = torch.exp(-(delt_m1_m2 - 0.0097) / (x_t2 + eps))
        ET1 = torch.exp(-(delt_m1_m2 - 0.0097) / (x_t1 + eps))
        Ed1 = torch.exp(-(delt_m0_m1) / (x_t1 + eps))
        Ed4 = torch.exp(-(delt_m3_m4) / (x_t1 + eps))
        Ed6 = torch.exp(-(delt_m5_m6) / (x_t1 + eps))
        Ed8 = torch.exp(-(delt_m7_m8) / (x_t1 + eps))
        Ed10 = torch.exp(-(delt_m9_m10) / (x_t1 + eps))
        Ed12 = torch.exp(-(delt_m11_m12) / (x_t1 + eps))
        Ed14 = torch.exp(-(delt_m13_end) / (x_t1 + eps))
        Eda = torch.exp(-(0.0097) / (x_t1 + eps))
        Edb = torch.exp(-(0.) / (x_t1 + eps))
        x_t1_star = x_t1 * (1 / (1 - x_t1 * torch.log(torch.cos(np.pi / 180 * flip_ang)) / esp))
        x_m0_star = x_m0 * (1 - torch.exp(-esp / (x_t1 + eps))) / (1 - torch.exp(-esp / (x_t1_star + eps)))
        Eetl = torch.exp(-etl / (x_t1_star + eps))
        Eesp = torch.exp(-esp / (x_t1 + eps))

        num_rep = 20 # number of repetitions to simulate to reach steady state
        m_current = x_m0                                                                                # M0
        for _ in range(num_rep):
            m_current = x_m0 * (1 - Ed1) + m_current * Ed1                                              # M1 (del_t = 0.0607)
            # m_current = m_current * ET2                                                               # M2, w/o b1 cor.
            t2_rad = np.pi / 2 * x_b1.to(x_t1.device)                                                   # M2, w/ b1 cor.
            m_current = m_current * (torch.sin(t2_rad) * torch.sin(t2_rad) * ET2 + \
                                    torch.cos(t2_rad) * torch.cos(t2_rad) * ET1)                        # M2, w/ b1 cor.
            m_current = x_m0 * (1 - Eda) + m_current * Eda                                              # M2 (del_t = 0.0097)
            # current_img_acq1 = m_current                                                                ### Acq1
            current_img_acq1 = m_current * torch.sin(np.pi / 180 * flip_ang)                            ### Acq1
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl                                       # M3 (del_t = 0.7296)
            m_current = x_m0 * (1 - Ed4) + m_current * Ed4                                              # M4 (del_t = 0.1221)
            m_current = -m_current * x_ie                                                               # M5
            m_current = x_m0 * (1 - Ed6) + m_current * Ed6                                              # M6 (del_t = 0.0355)
            m_current = x_m0 * (1 - Edb) + m_current * Edb                                              # M6 (del_t = 0)
            # current_img_acq2 = m_current                                                                ### Acq2
            current_img_acq2 = m_current * torch.sin(np.pi / 180 * flip_ang)                            ### Acq2
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl                                       # M7 (del_t = 0.7296)
            m_current = x_m0 * (1 - Ed8) + m_current * Ed8                                              # M8 (del_t = 0.1704)
            m_current = x_m0 * (1 - Edb) + m_current * Edb                                              # M8 (del_t = 0)
            # current_img_acq3 = m_current                                                                ### Acq3
            current_img_acq3 = m_current * torch.sin(np.pi / 180 * flip_ang)                            ### Acq3
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl                                       # M9 (del_t = 0.7296)
            m_current = x_m0 * (1 - Ed10) + m_current * Ed10                                            # M10 (del_t = 0.1704)
            m_current = x_m0 * (1 - Edb) + m_current * Edb                                              # M10 (del_t = 0)
            # current_img_acq4 = m_current                                                                ### Acq4
            current_img_acq4 = m_current * torch.sin(np.pi / 180 * flip_ang)                            ### Acq4
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl                                       # M11 (del_t = 0.7296)
            m_current = x_m0 * (1 - Ed12) + m_current * Ed12                                            # M12 (del_t = 0.1704)
            m_current = x_m0 * (1 - Edb) + m_current * Edb                                              # M12 (del_t = 0)
            # current_img_acq5 = m_current                                                                ### Acq5
            current_img_acq5 = m_current * torch.sin(np.pi / 180 * flip_ang)                            ### Acq5
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl                                       # M13 (del_t = 0.7296)
            m_current = x_m0 * (1 - Ed14) + m_current * Ed14                                            # M14

        # return torch.abs(current_img_acq1), torch.abs(current_img_acq2), torch.abs(current_img_acq3), torch.abs(current_img_acq4), torch.abs(current_img_acq5)
        return current_img_acq1, current_img_acq2, current_img_acq3, current_img_acq4, current_img_acq5

    def forward(
        self,
        init_map_t1: torch.Tensor,
        init_map_t2: torch.Tensor,
        init_map_pd: torch.Tensor,
        init_map_ie: torch.Tensor,
        init_map_b1: torch.Tensor,
    ) -> torch.Tensor:

        [init_img_acq1, init_img_acq2, init_img_acq3, init_img_acq4, init_img_acq5] = \
            self.qalas_forward_eq(init_map_t1, init_map_t2, init_map_pd, init_map_ie, init_map_b1)

        return init_img_acq1, init_img_acq2, init_img_acq3, init_img_acq4, init_img_acq5
