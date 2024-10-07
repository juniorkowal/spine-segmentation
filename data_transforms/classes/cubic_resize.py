import os

import h5py
import torch
import torch.nn.functional as F
import torchio as tio
from data_transforms.classes import PadToRatio
from utils.constants import COMPRESSION, TEST_SUBJECT


class CubicResize(tio.SpatialTransform):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            if tio.LABEL in image_name:
                resized_tensor = self.resize(image.data)
                image.set_data(resized_tensor.squeeze(0))
            else:
                resizer = tio.Resize(self.target_shape)
                resized_image_tensor = resizer(image.data)
                image.set_data(resized_image_tensor)
        return subject

    def resize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:# to ensure the tensor has 5 dimensions: (N, C, D, H, W)
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.float()
        unique_classes = torch.unique(tensor)
        unique_classes = unique_classes[unique_classes != 0]  # no background (0)
        
        _, _, D, H, W = tensor.shape
        target_D, target_H, target_W = self.target_shape
        target_tensor = torch.zeros((1, 1, target_D, target_H, target_W), dtype=torch.uint8)
        
        temp_h5_path = 'temp_resize.h5'
        
        with h5py.File(temp_h5_path, 'w') as h5f:
            for class_value in unique_classes:
                class_mask = (tensor == class_value).float()
                h5f.create_dataset(f'class_{int(class_value.item())}', data=class_mask.squeeze(0), compression=COMPRESSION)
        
        with h5py.File(temp_h5_path, 'r') as h5f:
            for class_value in unique_classes:
                class_mask = torch.tensor(h5f[f'class_{int(class_value.item())}'][:])
                class_mask = class_mask.unsqueeze(0)#.unsqueeze(0)
                
                resized_mask = F.interpolate(class_mask, size=(target_D, target_H, target_W), mode='trilinear', align_corners=False)
                resized_mask = (resized_mask > 0.5).byte()
                
                target_tensor = torch.max(target_tensor, resized_mask * class_value.byte())
        
        os.remove(temp_h5_path)
        return target_tensor


if __name__ == "__main__":
    shape = (64, 64, 128)
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        PadToRatio(ratio=(1, 1, 2), padding_mode=-1024),
        CubicResize(target_shape=shape),
        # LabelToHeatmap(sigma=1.0, threshold_value=0.5, mode='mask', 
        #                blur_output=1.0
        #                )
    ])
    
    # TEST_SUBJECT.plot()
    transformed_subject = transform(TEST_SUBJECT)
    transformed_subject.plot()
