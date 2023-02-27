"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import fastmri
import numpy as np
import torch

from .subsample import MaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )


class QALASSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        img_sense: SENSE reconstructed images.
        coil_sens: The coil sensitivity maps estimated with ESPIRiT.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace_acq1: torch.Tensor
    masked_kspace_acq2: torch.Tensor
    masked_kspace_acq3: torch.Tensor
    masked_kspace_acq4: torch.Tensor
    masked_kspace_acq5: torch.Tensor
    mask_acq1: torch.Tensor
    mask_acq2: torch.Tensor
    mask_acq3: torch.Tensor
    mask_acq4: torch.Tensor
    mask_acq5: torch.Tensor
    mask_brain: torch.Tensor
    # coil_sens: torch.Tensor
    b1: torch.Tensor
    ie: torch.Tensor
    num_low_frequencies: Optional[int]
    target_t1: torch.Tensor
    target_t2: torch.Tensor
    target_pd: torch.Tensor
    fname: str
    slice_num: int
    max_value_t1: float
    max_value_t2: float
    max_value_pd: float
    crop_size: Tuple[int, int]


class QALASDataTransform:
    """
    Data Transformer for training QALAS models.
    """

    def __init__(self, mask_func_acq1: Optional[MaskFunc] = None, mask_func_acq2: Optional[MaskFunc] = None, mask_func_acq3: Optional[MaskFunc] = None, \
                mask_func_acq4: Optional[MaskFunc] = None, mask_func_acq5: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func_acq1 = mask_func_acq1
        self.mask_func_acq2 = mask_func_acq2
        self.mask_func_acq3 = mask_func_acq3
        self.mask_func_acq4 = mask_func_acq4
        self.mask_func_acq5 = mask_func_acq5
        self.use_seed = use_seed

    def __call__(
        self,
        kspace_acq1: np.ndarray,
        kspace_acq2: np.ndarray,
        kspace_acq3: np.ndarray,
        kspace_acq4: np.ndarray,
        kspace_acq5: np.ndarray,
        mask_acq1: np.ndarray,
        mask_acq2: np.ndarray,
        mask_acq3: np.ndarray,
        mask_acq4: np.ndarray,
        mask_acq5: np.ndarray,
        mask_brain: np.ndarray,
        # coil_sens: np.ndarray,
        b1: np.ndarray,
        ie: np.ndarray,
        target_t1: Optional[np.ndarray],
        target_t2: Optional[np.ndarray],
        target_pd: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> QALASSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A QALASSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target_t1 is not None:
            target_torch_t1 = to_tensor(target_t1)
            target_torch_t2 = to_tensor(target_t2)
            target_torch_pd = to_tensor(target_pd)
            max_value_t1 = attrs["max_t1"]
            max_value_t2 = attrs["max_t2"]
            max_value_pd = attrs["max_pd"]
        else:
            target_torch_t1 = torch.tensor(0)
            target_torch_t2 = torch.tensor(0)
            target_torch_pd = torch.tensor(0)
            max_value_t1 = 0.0
            max_value_t2 = 0.0
            max_value_pd = 0.0

        kspace_torch_acq1 = to_tensor(kspace_acq1)
        kspace_torch_acq2 = to_tensor(kspace_acq2)
        kspace_torch_acq3 = to_tensor(kspace_acq3)
        kspace_torch_acq4 = to_tensor(kspace_acq4)
        kspace_torch_acq5 = to_tensor(kspace_acq5)
        mask_brain_torch = to_tensor(mask_brain)
        # coil_sens_torch = to_tensor(coil_sens)
        b1_torch = to_tensor(b1)
        ie_torch = to_tensor(ie)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func_acq1 is not None:
            masked_kspace_acq1, mask_torch_acq1, num_low_frequencies = apply_mask(
                kspace_torch_acq1, self.mask_func_acq1, seed=seed, padding=(acq_start, acq_end)
            )
            masked_kspace_acq2, mask_torch_acq2, num_low_frequencies = apply_mask(
                kspace_torch_acq2, self.mask_func_acq2, seed=seed, padding=(acq_start, acq_end)
            )
            masked_kspace_acq3, mask_torch_acq3, num_low_frequencies = apply_mask(
                kspace_torch_acq3, self.mask_func_acq3, seed=seed, padding=(acq_start, acq_end)
            )
            masked_kspace_acq4, mask_torch_acq4, num_low_frequencies = apply_mask(
                kspace_torch_acq4, self.mask_func_acq4, seed=seed, padding=(acq_start, acq_end)
            )
            masked_kspace_acq5, mask_torch_acq5, num_low_frequencies = apply_mask(
                kspace_torch_acq5, self.mask_func_acq5, seed=seed, padding=(acq_start, acq_end)
            )

            sample = QALASSample(
                masked_kspace_acq1=masked_kspace_acq1,
                masked_kspace_acq2=masked_kspace_acq2,
                masked_kspace_acq3=masked_kspace_acq3,
                masked_kspace_acq4=masked_kspace_acq4,
                masked_kspace_acq5=masked_kspace_acq5,
                mask_acq1=mask_torch_acq1.to(torch.bool),
                mask_acq2=mask_torch_acq2.to(torch.bool),
                mask_acq3=mask_torch_acq3.to(torch.bool),
                mask_acq4=mask_torch_acq4.to(torch.bool),
                mask_acq5=mask_torch_acq5.to(torch.bool),
                mask_brain=mask_brain_torch,
                # coil_sens=coil_sens_torch,
                b1=b1_torch,
                ie=ie_torch,
                num_low_frequencies=num_low_frequencies,
                target_t1=target_torch_t1,
                target_t2=target_torch_t2,
                target_pd=target_torch_pd,
                fname=fname,
                slice_num=slice_num,
                max_value_t1=max_value_t1,
                max_value_t2=max_value_t2,
                max_value_pd=max_value_pd,
                crop_size=crop_size,
            )
        else:
            masked_kspace_acq1 = kspace_torch_acq1
            masked_kspace_acq2 = kspace_torch_acq2
            masked_kspace_acq3 = kspace_torch_acq3
            masked_kspace_acq4 = kspace_torch_acq4
            masked_kspace_acq5 = kspace_torch_acq5
            shape = np.array(kspace_torch_acq1.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch_acq1 = torch.from_numpy(mask_acq1.reshape(*mask_shape).astype(np.float32))
            mask_torch_acq1 = mask_torch_acq1.reshape(*mask_shape)
            mask_torch_acq1[:, :, :acq_start] = 0
            mask_torch_acq1[:, :, acq_end:] = 0
            mask_torch_acq2 = torch.from_numpy(mask_acq2.reshape(*mask_shape).astype(np.float32))
            mask_torch_acq2 = mask_torch_acq2.reshape(*mask_shape)
            mask_torch_acq2[:, :, :acq_start] = 0
            mask_torch_acq2[:, :, acq_end:] = 0
            mask_torch_acq3 = torch.from_numpy(mask_acq3.reshape(*mask_shape).astype(np.float32))
            mask_torch_acq3 = mask_torch_acq3.reshape(*mask_shape)
            mask_torch_acq3[:, :, :acq_start] = 0
            mask_torch_acq3[:, :, acq_end:] = 0
            mask_torch_acq4 = torch.from_numpy(mask_acq4.reshape(*mask_shape).astype(np.float32))
            mask_torch_acq4 = mask_torch_acq4.reshape(*mask_shape)
            mask_torch_acq4[:, :, :acq_start] = 0
            mask_torch_acq4[:, :, acq_end:] = 0
            mask_torch_acq5 = torch.from_numpy(mask_acq5.reshape(*mask_shape).astype(np.float32))
            mask_torch_acq5 = mask_torch_acq5.reshape(*mask_shape)
            mask_torch_acq5[:, :, :acq_start] = 0
            mask_torch_acq5[:, :, acq_end:] = 0

            sample = QALASSample(
                masked_kspace_acq1=masked_kspace_acq1,
                masked_kspace_acq2=masked_kspace_acq2,
                masked_kspace_acq3=masked_kspace_acq3,
                masked_kspace_acq4=masked_kspace_acq4,
                masked_kspace_acq5=masked_kspace_acq5,
                mask_acq1=mask_torch_acq1.to(torch.bool),
                mask_acq2=mask_torch_acq2.to(torch.bool),
                mask_acq3=mask_torch_acq3.to(torch.bool),
                mask_acq4=mask_torch_acq4.to(torch.bool),
                mask_acq5=mask_torch_acq5.to(torch.bool),
                mask_brain=mask_brain_torch,
                # coil_sens=coil_sens_torch,
                b1=b1_torch,
                ie=ie_torch,
                num_low_frequencies=0,
                target_t1=target_torch_t1,
                target_t2=target_torch_t2,
                target_pd=target_torch_pd,
                fname=fname,
                slice_num=slice_num,
                max_value_t1=max_value_t1,
                max_value_t2=max_value_t2,
                max_value_pd=max_value_pd,
                crop_size=crop_size,
            )

        return sample
