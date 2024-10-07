from typing import Optional

import torch
import torchio as tio
from utils.constants import HEATMAP
from utils.misc import (apply_smooth_values, compute_bbox3d, convex_hull_mask,
                        create_heatmap, create_sphere_mask,
                        recalculate_centroids)


class LabelToHeatmap(tio.LabelTransform):
    r"""
    Transforms a label map into a heatmap.

    This transformation generates a heatmap from a label map using various modes
    such as Gaussian, distance-based, estimation-based, or precise heatmap generation.

    Args:
        sigma (Optional[float]): Standard deviation for the Gaussian kernel in 'gaussian' mode.
            This value must be provided if `mode` is set to 'gaussian'.
        threshold_value (Optional[float]): Threshold value for 'distance' mode.
            This value must be provided if `mode` is set to 'distance'.
        mode (str): Mode of heatmap generation. Options are:
            'gaussian': Uses a fixed Gaussian kernel.
            'distance': First calculates bboxes encompassing each label,
                          then creates mask by placing spheres in place of each bbox,
                          filling it. Mask is then treated by ndi.distance_transform_edt to create
                          smooth transition between values with respect to provided threshold_value.
            'estimate': Estimates sigma from bounding boxes.
            'precise': First estimates sigma from bounding boxes,
                         then looks for optimal sigma that encompasses each label,
                         by iterating over list of sigmas and thresholding. List is
                         created from estimated sigma to one 30% larger.

            Default is 'gaussian'.
        maximum (bool): Whether to take the maximum value across channels for the heatmap.
            This option is not compatible with 'distance' mode. Default is False.
        blur_output (Optional[float]): Amount of blurring to apply to the output heatmap.
            If not provided, no blurring is applied.
        multichannel (bool): Whether to generate a multi-channel heatmap.
            This option is not compatible with 'distance' mode. Default is False.

    Example:
        >>> import torchio as tio
        >>> transform = LabelToHeatmap(sigma=2.0, mode='gaussian', blur_output=1.0)
        >>> transformed_subject = transform(subject)
    """
    def __init__(self, sigma: Optional[float] = None,
                 threshold_value: Optional[float] = None,
                 mode: str = 'gaussian',
                 maximum: bool = False,
                 blur_output: Optional[float] = None,
                 multichannel: bool = False):
        super(LabelToHeatmap, self).__init__()
        self.sigma = sigma
        self.threshold_value = threshold_value
        self.mode = mode
        self.maximum = maximum
        self.blur_output = blur_output
        self.multichannel = multichannel

        assert self.mode in ['gaussian', 'distance', 'estimate', 'precise', 'convex_hull', 'mask'], "Mode must be one of 'gaussian', 'distance', 'estimate', 'precise'."

        if self.mode == 'distance':
            assert not self.multichannel, "multichannel is not compatible with mode 'distance'."
            assert not self.maximum, "maximum is not compatible with mode 'distance'."

        if self.mode in ['distance', 'estimate', 'precise']:
            assert self.threshold_value is not None, f"threshold_value for '{self.mode}' mode can't be None."

        if self.mode == 'gaussian':
            assert sigma is not None, "Gaussian mode needs to have sigma value provided."

    def apply_transform(self, subject):
        label_map = next(image for image in subject.get_images(intensity_only=False) if image[tio.TYPE] == tio.LABEL)
        self.shape = subject.spatial_shape

        ctds = recalculate_centroids(label_map.data.squeeze())
        heatmap = self.generate_heatmap(ctds, label_map.data.squeeze())

        if torch.isnan(heatmap).any():
            heatmap = torch.zeros_like(heatmap, dtype=torch.float32)

        if heatmap.dim() < 4:
            heatmap = heatmap.unsqueeze(0)

        subject.add_image(tio.LabelMap(tensor=heatmap), HEATMAP)
        del heatmap

        return subject

    def generate_heatmap(self, ctds, mask):
        heatmap = torch.zeros(*self.shape, dtype=torch.float32)
        labels = [int(sublist[0]) for sublist in ctds if sublist]
        heatmaps = torch.zeros(len(ctds), *self.shape, dtype=torch.float32)

        for i, label in enumerate(labels):
            if self.mode in ['estimate', 'precise']:
                bbox = self.get_label_bbox(mask, label)
                sigma = self.estimate_sigma_from_bbox(bbox)
                if self.mode == 'precise':
                    heatmap_op = self.find_optimal_sigma(mask.clone(), [ctds[i]], label, sigma)
                else:
                    heatmap_op = create_heatmap([ctds[i]], sigma, self.shape)
            elif self.mode == 'gaussian':
                heatmap_op = create_heatmap([ctds[i]], self.sigma, self.shape)
            elif self.mode == 'distance':
                bboxes_list = [self.get_label_bbox(mask, label) for label in labels]
                sphere_mask = create_sphere_mask(bboxes_list, mask.shape)

                heatmap = torch.from_numpy(apply_smooth_values(sphere_mask, threshold=self.threshold_value, decay_rate=0.1))
                break
            elif self.mode == 'convex_hull':
                sphere_mask = convex_hull_mask(mask.squeeze().numpy()>0.5)
                # tio.Subject(img = tio.LabelMap(tensor=torch.from_numpy(sphere_mask).unsqueeze(0))).plot()
                heatmap = torch.from_numpy(apply_smooth_values(sphere_mask, threshold=self.threshold_value, decay_rate=0.1))
                break
            elif self.mode == 'mask':
                sphere_mask = mask.squeeze().numpy()>0.5
                heatmap = torch.from_numpy(apply_smooth_values(sphere_mask, threshold=self.threshold_value, decay_rate=0.1))
                break

            if self.multichannel or self.maximum:
                heatmaps[i] += heatmap_op
            else:
                heatmap += heatmap_op

        if self.maximum:
            heatmap, _ = torch.max(heatmaps, dim=0)
        if self.multichannel:
            heatmap = heatmaps

        if self.blur_output is not None:
            blur = tio.Blur(self.blur_output)
            heatmap = blur(heatmap.unsqueeze(0)).squeeze() if heatmap.dim() < 4 else blur(heatmap).squeeze()

        heatmap /= torch.max(heatmap)
        return heatmap

    def get_label_bbox(self, mask, label):
        mask_label = mask == label
        bbox = compute_bbox3d(mask_label, self.threshold_value)
        return bbox

    def estimate_sigma_from_bbox(self, bbox):
        min_coords = torch.tensor(bbox[0])
        max_coords = torch.tensor(bbox[1])
        bbox_size = max_coords - min_coords
        avg_side_length = torch.mean(bbox_size.float()).item()
        sigma = avg_side_length / (2.0 * torch.sqrt(torch.log(1.0 / torch.tensor(self.threshold_value))))
        return sigma

    def find_optimal_sigma(self, mask, ctds, label, starting_sigma):
        mask[mask!=label] = 0
        sigmas = torch.arange(starting_sigma, starting_sigma * 1.3)
        heatmap = create_heatmap(ctds, starting_sigma, self.shape)
        for sigma in sigmas:
            heatmap = create_heatmap(ctds, sigma, self.shape)
            thresholded_heatmap = heatmap > self.threshold_value

            if torch.all(thresholded_heatmap[mask > 0]):
                return heatmap

        return heatmap
