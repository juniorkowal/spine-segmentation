import random
from typing import Union

import torch
import torchio as tio


class MaximumProjection(tio.IntensityTransform):
    r"""Applies a maximum intensity projection (MIP) to an image along a specified dimension or across all dimensions.
    
    This transformation computes the maximum value projection of the image data over a specified
    number of slices along a specified dimension or across all dimensions, maintaining the original image dimensions.
    
    Args:
        dimension (Union[int, str]): The dimension along which to apply MIP (0 for depth, 1 for height, 2 for width, 'all' for all dimensions).
        If 'all' is provided, then output will have 3 channels, 1 for each MIP of corresponding dimension.
        If 'all+' is provided, then output will have 4 channels with original image tensor at the beginning.
        If 'all+r' is provided, then channels will be shuffled randomly afterwards.
        slices_num (int): The number of slices for the maximum intensity projection.
                          Defaults to 15.
        **kwargs: Additional keyword arguments for torchio.transforms.Transform.
    """
    
    def __init__(self, dimension: Union[int, str] = 2, slices_num: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.slices_num = slices_num
        self.args_names = ['dimension', 'slices_num']
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in self.get_images(subject):
            assert isinstance(image, tio.ScalarImage)
            self.apply_mip(image)
        return subject

    def apply_mip(self, image: tio.ScalarImage) -> None:
        tensor = image.data
        mip_tensors = []

        if self.dimension in ['all+', 'all+r']:
            mip_tensors.append(tensor.squeeze())  # include the original tensor

        if self.dimension in ['all', 'all+', 'all+r']:
            mip_tensors.extend([self.create_mip(tensor, self.slices_num, dim) for dim in range(3)])
        else:
            mip_tensors.append(self.create_mip(tensor, self.slices_num, self.dimension))

        if self.dimension == 'all+r':
            random.shuffle(mip_tensors)  # randomize the order of the tensors

        mip_tensor = torch.stack(mip_tensors, dim=0)
        image.set_data(mip_tensor)

    def create_mip(self, torch_img: torch.Tensor, slices_num: int, dimension: int) -> torch.Tensor:
        """
        Create the maximum intensity projection (MIP) image from the original image.
        
        Args:
            torch_img (torch.Tensor): The input image tensor of shape (1, D, H, W).
            slices_num (int): The number of slices for the maximum intensity projection.
            dimension (int): The dimension along which to apply MIP (0 for depth, 1 for height, 2 for width).
            
        Returns:
            torch.Tensor: The MIP image tensor with the same shape as the input.
        """
        torch_img.squeeze()
        
        img_shape = torch_img.shape
        torch_mip = torch.zeros_like(torch_img)

        for i in range(img_shape[dimension]):  # loop over the specified dimension
            start = max(0, i - slices_num)
            if dimension == 0:
                torch_mip[i, :, :] = torch.amax(torch_img[start:i + 1, :, :], dim=0)
            elif dimension == 1:
                torch_mip[:, i, :] = torch.amax(torch_img[:, start:i + 1, :], dim=1)
            elif dimension == 2:
                torch_mip[:, :, i] = torch.amax(torch_img[:, :, start:i + 1], dim=2)

        return torch_mip