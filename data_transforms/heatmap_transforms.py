from typing import Tuple

import torchio as tio
from data_transforms.classes import LabelToHeatmap, PadDimTo, PadToRatio, AlgorithmicDenoise, CubicResize
from utils.misc import transform_timeit


def heatmap_transforms(data_shape: Tuple[int, int, int]):
    """
    Creates preprocessing and postprocessing transformations for spine localization.

    Args:
        data_shape (Tuple[int, int, int]): Shape of the data size after resizing (e.g., (64, 64, 128)).

    Returns:
        Tuple[tio.Compose, tio.Compose, tio.Compose, tio.Compose, tio.Compose]:
        - spine_loc_prep: Preprocessing for parsing datasets (e.g., to h5).
        - spine_loc_train_post: Postprocessing for training datasets.
        - spine_loc_val_post: Postprocessing for validation datasets.
        - spine_loc_train: Transformations for raw training data.
        - spine_loc_val: Transformations for raw validation data.
    """
    spine_loc_prep = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1), # 8
        tio.Clamp(out_min = -1024),
        tio.Blur(std = 0.75), # SmoothingRecursiveGaussian(0.75)
        PadToRatio(ratio=(1,1,2)),
        tio.Resize(target_shape=data_shape),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    spine_loc_train_post = tio.Compose([
        tio.RandomAffine(scales=0.1, degrees=10),
        tio.RandomFlip(axes = (2), flip_probability=0.5),
        tio.RandomGamma(log_gamma=0.3),
        tio.RescaleIntensity(out_min_max=(0,1)),
        PadDimTo(data_shape),
        LabelToHeatmap(mode='distance', threshold_value=0.5, blur_output=1.0),
    ])

    spine_loc_val_post = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0,1)),
        PadDimTo(data_shape),
        LabelToHeatmap(mode='distance', threshold_value=0.5, blur_output=1.0),
    ])

    spine_loc_train = tio.Compose([
        spine_loc_prep,
        spine_loc_train_post
    ])

    spine_loc_val = tio.Compose([
        spine_loc_prep,
        spine_loc_val_post
    ])
    
    return  spine_loc_prep, spine_loc_train_post, spine_loc_val_post, spine_loc_train, spine_loc_val



def get_heatmap_prepro(data_shape):
    """Experimenting with different preprocessing techniques."""
    # payer c. kinda
    prep1 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.Clamp(out_min = -1024),
        tio.Blur(std = 0.75),
        PadToRatio(ratio=(1,1,2)),
        tio.Resize(target_shape=data_shape),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # with noise
    prep2 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        PadToRatio(ratio=(1,1,2)),
        tio.Resize(target_shape=data_shape),
        tio.RescaleIntensity(out_min_max=(0,1)),

    ])

    # nlm3denoise test
    prep3 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1), # 8
        PadToRatio(ratio=(1,1,2)),
        tio.Resize(target_shape=data_shape),
        AlgorithmicDenoise(),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # cubic resize test
    prep4 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.Clamp(out_min = -1024),
        tio.Blur(std = 0.75),
        PadToRatio(ratio=(1,1,2)),
        CubicResize(target_shape=data_shape),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    return prep1, prep2, prep3, prep4



def create_heatmap_transform(data_shape, heatmap_mode, sigma=None, blur_output=1.0, maximum=False):
    """Create a transformation pipeline for the heatmap."""
    common_transforms = [
        tio.RandomAffine(scales=0.1, degrees=10),
        tio.RandomFlip(axes=(2), flip_probability=0.5),
        tio.RandomGamma(log_gamma=0.3),
        tio.RescaleIntensity(out_min_max=(0,1)),
        PadDimTo(data_shape)
    ]
    
    heatmap_transform = LabelToHeatmap(
        mode=heatmap_mode, 
        sigma=sigma, 
        threshold_value=0.5, 
        maximum=maximum, 
        blur_output=blur_output
    )
    
    return tio.Compose(common_transforms + [heatmap_transform]), tio.Compose([
        tio.RescaleIntensity(out_min_max=(0,1)),
        PadDimTo(data_shape),
        heatmap_transform
    ])

def get_heatmap_postpro(data_shape):
    """Create multiple transformation pipelines for different heatmap modes."""
    
    prep1 = create_heatmap_transform(data_shape, 'gaussian', sigma=15, blur_output=3.0, maximum=True)
    prep2 = create_heatmap_transform(data_shape, 'distance', blur_output=1.0)
    prep3 = create_heatmap_transform(data_shape, 'estimate', blur_output=1.0, maximum=True)
    prep4 = create_heatmap_transform(data_shape, 'precise', blur_output=1.0, maximum=True)
    prep5 = create_heatmap_transform(data_shape, 'convex_hull', blur_output=1.0)
    prep6 = create_heatmap_transform(data_shape, 'mask', blur_output=1.0)

    return prep1, prep2, prep3, prep4, prep5, prep6
