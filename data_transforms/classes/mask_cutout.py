from typing import Optional, Tuple, Union

import torch
import torchio as tio
from utils.constants import HEATMAP, TypeTripletInt
from utils.misc import compute_bbox3d


class MaskCutout(tio.SpatialTransform):
    r"""Applies a cutout transformation to an image and its label based on a mask.

    This transformation uses either a label or a heatmap to compute a bounding box
    and cutout the corresponding region from the image and its label.

    Args:
        threshold (Optional[Union[str, int]]): The threshold used to compute the 
            bounding box for the cutout. This could be a fixed value or a 
            method to determine the threshold dynamically.
        mode (str, optional): Specifies the mode of cutout. Options are:
            - 'label': Use the label to compute the bounding box.
            - 'heatbox': Use a heatmap to compute the bounding box.
            Default is 'label'.
        **kwargs: Additional keyword arguments for the parent class.
            See :class:`~torchio.transforms.SpatialTransform` for details.

    Example:
        >>> import torchio as tio
        >>> transform = MaskCutout(threshold=0.5, mode='heatbox')
        >>> transformed_subject = transform(subject)
    """

    def __init__(self, threshold: Optional[Union[str, int]] = None, mode: str = 'label', **kwargs):
        super().__init__(**kwargs)
        assert mode in ['label', 'heatbox'], "Mode must be either 'label' or 'heatbox'"
        self.threshold = threshold
        self.mode = mode
        self.args_names = ['threshold', 'mode']

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Applies the cutout transformation to the image and label.

        Depending on the mode, this method will either use a label or a heatmap
        to compute the bounding box and apply the cutout.

        Args:
            subject (tio.Subject): A TorchIO Subject containing the image,
                label, and optionally a heatmap.

        Returns:
            tio.Subject: The subject with the cutout applied.
        """
        image = subject[tio.IMAGE]
        label = subject[tio.LABEL]

        if self.mode == 'heatbox':
            heatmap = subject[HEATMAP]
            bbox = compute_bbox3d(heatmap.data.squeeze(), self.threshold)
            image.set_data(self.cutout_bbox(image, bbox))
            label.set_data(self.cutout_bbox(label, bbox))
            del subject[HEATMAP]
        elif self.mode == 'label':
            bbox = compute_bbox3d(label.data.squeeze())
            image.set_data(self.cutout_bbox(image, bbox))
            label.set_data(self.cutout_bbox(label, bbox))

        return subject

    def cutout_bbox(self, volume: tio.Image, bbox: Tuple[TypeTripletInt, TypeTripletInt]) -> torch.Tensor:
        """Extracts the cutout region from the volume based on the bounding box.

        Args:
            volume (tio.Image): The image or label volume to apply the cutout.
            bbox (Tuple[Tuple[int, int, int], Tuple[int, int, int]]): The bounding 
                box defining the region to be cut out.

        Returns:
            torch.Tensor: The volume with the cutout region applied.
        """
        volume_data = volume.data.squeeze()
        ((start_x, start_y, start_z), (end_x, end_y, end_z)) = bbox
        cutout = volume_data[start_x:end_x + 1, start_y:end_y + 1, start_z:end_z + 1]
        return cutout.unsqueeze(0)

