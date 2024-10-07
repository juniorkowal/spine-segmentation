import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio


class SmoothingRecursiveGaussian(tio.IntensityTransform):
    r"""Blur an image using a Smoothing Recursive Gaussian filter from SimpleITK.

    Args:
        std: float :math:`(\sigma)` representing the
            the standard deviation (in mm) of the Gaussian kernel used to
            blur the image.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    
    def __init__(
        self,
        std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.std = std
        self.args_names = ['std']

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        stds = self.std, self.std, self.std
        for image in self.get_images_dict(subject).values():
            repeats = image.num_channels, 1
            std_channels: np.ndarray
            std_channels = np.tile(stds, repeats)  # type: ignore[arg-type]
            transformed_tensors = []
            for stds, channel in zip(std_channels, image.data):
                transformed_tensor = sitk.GetImageFromArray(channel.numpy())
                transformed_tensor = sitk.SmoothingRecursiveGaussian(transformed_tensor, stds)
                transformed_tensor = sitk.GetArrayFromImage(transformed_tensor)
                transformed_tensors.append(torch.from_numpy(transformed_tensor))
            image.set_data(torch.stack(transformed_tensors))
        return subject
