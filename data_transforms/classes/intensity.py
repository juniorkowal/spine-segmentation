import random
from typing import Tuple

import torch
import torchio as tio


class ShiftScale(tio.IntensityTransform):
    r"""Applies a shift and scale transformation to an image, with added random variations.

    This transformation modifies the image intensity by first applying a fixed shift and scale,
    then introducing randomness through additional shifts and scaling factors that are uniformly sampled.

    Args:
        shift (float): Constant value added to all pixels in the image, shifting their intensity.
        scale (float): Constant factor by which all pixel values are multiplied, scaling their intensity.
        random_shift (float): Maximum magnitude of additional random shift applied to pixels. 
                              The random shift values are sampled uniformly from [-random_shift, random_shift].
        random_scale (float): Maximum magnitude of additional random scaling factor applied to pixels.
                              The random scale factors are sampled uniformly from [-random_scale, random_scale].
        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.

    The operations are performed in the following order:
        1. image += shift  # Apply fixed shift
        2. image *= scale  # Apply fixed scale
        3. image += random float uniformly sampled from [-random_shift, random_shift]  # Apply random shift
        4. image *= 1 + random float uniformly sampled from [-random_scale, random_scale]  # Apply random scale

    Example:
        >>> import torchio as tio
        >>> transform = ShiftScale(shift=10, scale=1.5, random_shift=5, random_scale=0.2)
        >>> transformed = transform(subject)  # 'subject' is an instance of tio.Subject

    """
    
    def __init__(
        self,
        shift: float,
        scale: float,
        random_shift: float,
        random_scale: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shift, self.scale = shift, scale
        self.random_shift, self.random_scale = random_shift, random_scale

        self.args_names = ['shift', 'scale', 'random_shift', 'random_scale']

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in self.get_images(subject):
            assert isinstance(image, tio.ScalarImage)
            self.apply_shiftscale(image)
        return subject

    def apply_shiftscale(self, image: tio.ScalarImage) -> None:
        image.set_data(self.shiftscale(image.data))

    def shiftscale(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.FloatTensor):
            tensor = tensor.to(torch.float32)
        tensor += self.shift
        tensor *= self.scale
        tensor += torch.empty(1).uniform_(-self.random_shift, self.random_shift)
        tensor *= 1 + torch.empty(1).uniform_(-self.random_scale, self.random_scale)
        return tensor



class GammaUnnormalized(tio.IntensityTransform):
    r"""Applies a gamma correction transformation to an image.

    This transformation modifies the image intensity by applying a gamma correction
    factor `gamma` directly on the unnormalized image data and then scaling it back to 
    the original range.

    Args:
        gamma_range Tuple(float, float): Gamma range to uniformly select gamma correction factor to be applied.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, gamma_range: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.gamma_range = gamma_range
        self.args_names = ['gamma_range']

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in self.get_images(subject):
            assert isinstance(image, tio.ScalarImage)
            self.apply_gamma_correction(image)
        return subject

    def apply_gamma_correction(self, image: tio.ScalarImage) -> None:
        tensor = image.data
        tensor = self.change_gamma_unnormalized(tensor)
        image.set_data(tensor)

    def change_gamma_unnormalized(self, img_tensor: torch.Tensor) -> torch.Tensor:
        min_value = img_tensor.min()
        max_value = img_tensor.max()

        normalized = (img_tensor - min_value) / (max_value - min_value)
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        gamma_corrected = torch.pow(normalized, gamma)
        rescaled = gamma_corrected * (max_value - min_value) + min_value
        return rescaled