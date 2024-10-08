import time
from typing import Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import torch
from utils.constants import TypeTripletInt
import scipy
import torch.nn as nn


def otsu_torch(tensor: torch.Tensor) -> float:
    n = 256
    hist = torch.histc(tensor, bins=n, min=torch.min(tensor), max=torch.max(tensor))
    bin_edges = torch.linspace(torch.min(tensor), torch.max(tensor), steps=n+1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    w1 = torch.cumsum(hist, dim=0)
    w2 = torch.cumsum(hist.flip(0), dim=0).flip(0)

    mu1 = torch.cumsum(hist * bin_centers, dim=0) / w1
    mu2 = (torch.cumsum((hist * bin_centers).flip(0), dim=0) / w2.flip(0)).flip(0)

    sigma_total = w1[:-1] * w2[1:] * (mu1[:-1] - mu2[1:]) ** 2

    max_val_index = torch.argmax(sigma_total)
    t = bin_centers[:-1][max_val_index].item()
    return t


def compute_bbox3d(tensor: torch.Tensor, threshold: Union[float, str] = 0.5, format: str = 'min_max') -> Tuple[TypeTripletInt, TypeTripletInt]:
    """
    Computes the 3D bounding box of a torch tensor after thresholding.

    Args:
        tensor (torch.Tensor): 3D tensor to compute bounding box from.
        threshold (float): Threshold value for binarizing the tensor.
        format (str, optional): Format of the bounding box. 
            - 'min_max': Returns ((min x, min y, min z), (max x, max y, max z))
            - 'center_whd': Returns ((center x, center y, center z), (width, height, depth)). 
            Defaults to 'min_max'.

    Returns:
        - If format='min_max': Tuple[Tuple[int, int, int], Tuple[int, int, int]]
          Bounding box coordinates in (min coords, max coords) format.
        - If format='center_whd': Tuple[Tuple[int, int, int], Tuple[int, int, int]]
          Bounding box coordinates in (center, width, height, depth) format.

    Raises:
        ValueError: If an unsupported format is specified.

    Notes:
        - The 'min_max' format gives exact integer coordinates of the bounding box.
        - The 'center_whd' format computes the center of the bounding box as float values.
        - If no non-zero elements are found after thresholding, returns zeroed bounding box coordinates.
    """
    dims = tensor.dim()
    assert dims == 3, f"Data should be 3D, not {dims}D"
    
    if isinstance(threshold, str):
        threshold = otsu_torch(tensor)
        
    tensor_thresholded = torch.where(tensor > threshold, torch.tensor(1), torch.tensor(0))
    nonzero_indices = torch.nonzero(tensor_thresholded, as_tuple=False)
    
    if len(nonzero_indices) == 0:
        bbox = (0, 0, 0, 0, 0, 0)
    else:
        min_coords = torch.min(nonzero_indices, dim=0)[0]
        max_coords = torch.max(nonzero_indices, dim=0)[0]
        
        if format == 'min_max':
            bbox = (tuple(min_coords.numpy()), tuple(max_coords.numpy()))
        elif format == 'center_whd':
            center = (min_coords + max_coords) / 2.0
            width = max_coords[0] - min_coords[0] + 1.0
            height = max_coords[1] - min_coords[1] + 1.0
            depth = max_coords[2] - min_coords[2] + 1.0
            bbox = (tuple(center.int().numpy()), (width.int().item(), height.int().item(), depth.int().item()))
        else:
            raise ValueError("Unsupported bbox format. Please use 'min_max' or 'center_whd'.")
    
    return bbox


def create_bbox_image(bbox: Tuple[Tuple[int, int, int], Tuple[int, int, int]], shape: Tuple[int, int, int], thickness: int = 1) -> torch.Tensor:
    """
    Create a 3D tensor with 1s representing the bounding box edges and 0s elsewhere.

    Args:
        bbox (Tuple[Tuple[int, int, int], Tuple[int, int, int]]): The bounding box defined by (min_coords, max_coords).
        shape (Tuple[int, int, int]): The shape of the output tensor.
        thickness (int, optional): Thickness of the bounding box edges. Defaults to 1.

    Returns:
        torch.Tensor: A tensor with 1s along the edges of the bounding box.
    """
    bbox_image = torch.zeros(shape, dtype=torch.uint8)

    min_coords, max_coords = bbox[0], bbox[1]

    x_indices = torch.arange(min_coords[0], max_coords[0])
    y_indices = torch.arange(min_coords[1], max_coords[1])
    z_indices = torch.arange(min_coords[2], max_coords[2])

    bbox_image[min_coords[0]:min_coords[0] + thickness, y_indices, z_indices[:, None]] = 1
    bbox_image[max_coords[0] - thickness:max_coords[0], y_indices, z_indices[:, None]] = 1

    bbox_image[x_indices, min_coords[1]:min_coords[1] + thickness, z_indices[:, None]] = 1
    bbox_image[x_indices, max_coords[1] - thickness:max_coords[1], z_indices[:, None]] = 1

    bbox_image[x_indices, y_indices[:, None], min_coords[2]:min_coords[2] + thickness] = 1
    bbox_image[x_indices, y_indices[:, None], max_coords[2] - thickness:max_coords[2]] = 1

    return bbox_image


def combine_bboxes_image(bboxes: list):
    """Combine bounding boxes images into a single image tensor."""
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    combined = torch.zeros_like(bboxes[0]) if bboxes else None
    for bbox in bboxes:
        combined = torch.where(combined == 0, bbox, combined)
    return combined


def combine_bboxes(bboxes):
    """
    Combine multiple bounding boxes coordinates into one by finding the minimum and maximum coordinates.

    Args:
    bboxes (list of tuples): List of bounding boxes, where each bounding box is represented
                             as a tuple (min_coords, max_coords).

    Returns:
    tuple: Combined bounding box represented as a tuple (min_combined_coords, max_combined_coords).
    """
    if not bboxes:
        raise ValueError("The list of bounding boxes is empty")

    # Initialize min_combined and max_combined with the first bounding box
    min_combined, max_combined = bboxes[0]

    for bbox in bboxes[1:]:
        min_coords, max_coords = bbox
        min_combined = tuple(min(mc, c) for mc, c in zip(min_combined, min_coords))
        max_combined = tuple(max(mc, c) for mc, c in zip(max_combined, max_coords))

    return min_combined, max_combined

def convex_hull_mask(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    """

    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"
    
    points = np.argwhere(image).astype(np.int16)
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    idx_2d = np.indices(image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d
    
    mask = np.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:,:,0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask

def create_sphere_mask(bboxes, shape):
    """Create a binary mask with spheres for each bounding box."""
    mask = np.zeros(shape, dtype=bool)
    epsilon = 1e-10
    
    for bbox in bboxes:
        min_coords, max_coords = bbox
        min_coords = np.array(min_coords)
        max_coords = np.array(max_coords)
        center = (min_coords + max_coords) // 2
        radii = (max_coords - min_coords) / 2.0

        radii = np.maximum(radii, epsilon)

        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]

        distance = np.sqrt(((zz - center[0]) / radii[0]) ** 2 +
                           ((yy - center[1]) / radii[1]) ** 2 +
                           ((xx - center[2]) / radii[2]) ** 2)
        
        mask[distance <= 1] = True
        
    return mask

def apply_smooth_values(sphere_mask, threshold=0.5, decay_rate=0.1):
    """Apply smooth values inside and outside the spheres in the binary mask."""
    shape = sphere_mask.shape
    
    heatmap = np.zeros(shape, dtype=np.float32)

    distance_inside = ndi.distance_transform_edt(sphere_mask)
    smooth_inside = (distance_inside / distance_inside.max())
    smooth_inside = threshold + smooth_inside * (1 - threshold)
    heatmap[sphere_mask] = smooth_inside[sphere_mask]
    
    distance_outside = ndi.distance_transform_edt(~sphere_mask)
    smooth_outside = np.exp(-decay_rate * distance_outside)
    
    smooth_outside = smooth_outside * threshold
    heatmap[~sphere_mask] = smooth_outside[~sphere_mask]
    
    return heatmap


def gaussian(xL, yL, zL, H, W, D, sigma):
    r, c, t = torch.meshgrid(torch.arange(H), torch.arange(W), torch.arange(D), indexing='ij')
    exponent = -((c - xL) ** 2 + (r - yL) ** 2 + (t - zL) ** 2) / (2 * sigma ** 2)
    return torch.exp(exponent)

def create_heatmap(ctds, sigma, shape):
    heatmap = torch.zeros(shape, dtype=torch.float32)
    for label, coordy, coordx, coordz in ctds:
        channel = gaussian(coordx, coordy, coordz, *shape, sigma)
        heatmap += channel
    return heatmap

def recalculate_centroids(mask):
    labels = torch.unique(mask)[1:]
    ctds = []
    for label in labels:
        mask_temp = mask == label
        ctd = center_of_mass(mask_temp)
        ctds.append([label.item()] + [round(x.item(), 1) for x in ctd])
    return ctds

def center_of_mass(mask):
    grids = torch.meshgrid([torch.arange(dim, dtype=torch.float32) for dim in mask.shape], indexing='ij')
    normalizer = mask.sum()
    weighted_sums = [torch.sum(mask * grid) for grid in grids]
    centroid = [sum_value / normalizer for sum_value in weighted_sums]
    return centroid


def transform_timeit(transform, subject):
    start_time = time.time()
    subject = transform(subject)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Transform took: {elapsed_time} seconds")


def batchnorm_to_instance(model):
    batchnorm_layers = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            batchnorm_layers[name] = module
    
    for name, module in batchnorm_layers.items():
        parent = model
        for part in name.split('.')[:-1]:
            parent = getattr(parent, part)
        
        in_channels = module.num_features
        instance_norm = nn.InstanceNorm3d(in_channels, affine=True) # affine true or false?
        setattr(parent, name.split('.')[-1], instance_norm)

if __name__ == "__main__":
    ...
    # import numpy as np
    # import torchio as tio
    # from constants import TEST_SUBJECT
    # from data_transforms.label2heatmap import LabelToHeatmap
    # from data_transforms.torchio_fixes import TioResize

    # transform = tio.Compose([
    #     tio.ToCanonical(),
    #     tio.Resample(1),
    #     tio.RemapLabels(remapping={28:26}),
    #     TioResize(target_shape=(64,64,128)),
    #     tio.RescaleIntensity(out_min_max=(-1,1)),
    #     # LabelToHeatmap(sigma = np.arange(1,20,1), threshold_value=0.5)        
    # ])    
    # TEST_SUBJECT = transform(TEST_SUBJECT)
    # mask = TEST_SUBJECT[tio.LABEL][tio.DATA].squeeze()

    # print(compute_bbox3d(tensor=mask, threshold=0.5, format='center_whd'))
    # print(compute_bbox3d(tensor=mask, threshold=0.5, format='min_max'))

    # bbox = compute_bbox3d(tensor=mask, threshold=0.5, format='min_max')

    # bbox_image = create_bbox_image(bbox, TEST_SUBJECT.spatial_shape, thickness=5)

    # TEST_SUBJECT['bbox'] = tio.LabelMap(tensor = bbox_image.unsqueeze(0))

    # TEST_SUBJECT.plot()
