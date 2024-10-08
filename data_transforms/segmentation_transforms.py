from typing import Tuple

import torchio as tio
from data_transforms.classes import PadDimTo, MaskCutout, AlgorithmicDenoise
from utils.constants import remapping_binary
from utils.misc import transform_timeit


def binary_segmentation_transforms(data_shape: Tuple[int, int, int]):
    """
    Creates preprocessing and postprocessing transformations for spine segmentation.

    Args:
        data_shape (Tuple[int, int, int]): Shape of the data size after resizing (e.g., (64, 64, 128)).
        heatmap_sigma (float): Sigma value for generating heatmaps.

    Returns:
        Tuple[tio.Compose, tio.Compose, tio.Compose, tio.Compose, tio.Compose]:
        - spine_loc_prep: Preprocessing for parsing datasets (e.g., to h5).
        - spine_loc_train_post: Postprocessing for training datasets.
        - spine_loc_val_post: Postprocessing for validation datasets.
        - spine_loc_train: Transformations for raw training data.
        - spine_loc_val: Transformations for raw validation data.
    """
    spine_seg_prep = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1), # 8
        tio.Clamp(out_min = -1024),
        tio.Blur(std = 0.75), # SmoothingRecursiveGaussian(0.75)
        MaskCutout(threshold=0.5)
    ])

    spine_seg_train_post = tio.Compose([
        tio.RandomAffine(scales=0.1, degrees=10),
        tio.RandomFlip(axes = (2), flip_probability=0.5),
        tio.RandomGamma(log_gamma=0.3),
        tio.RescaleIntensity(out_min_max=(0,1)),
        tio.RemapLabels(remapping=remapping_binary),
        PadDimTo(size=data_shape),
    ])

    spine_seg_val_post = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0,1)), 
        tio.RemapLabels(remapping=remapping_binary),
        PadDimTo(size=data_shape),
    ])

    spine_seg_train = tio.Compose([
        spine_seg_prep,
        spine_seg_train_post
    ])

    spine_seg_val = tio.Compose([
        spine_seg_prep,
        spine_seg_val_post
    ])
    
    return  spine_seg_prep, spine_seg_train_post, spine_seg_val_post, spine_seg_train, spine_seg_val



def get_segmentation_prepro():
    # payer c. kinda
    prep1 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.Clamp(out_min = -1024),
        tio.Blur(std = 0.75),
        MaskCutout(threshold=0.5),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # with noise
    prep2 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        MaskCutout(threshold=0.5),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # nlm3denoise test
    prep4 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1), # 8
        tio.RescaleIntensity(out_min_max=(0,1)),
        MaskCutout(threshold=0.5),
        AlgorithmicDenoise(mode='nlm3d'),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # bm4d denoise
    prep5 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.RescaleIntensity(out_min_max=(0,1)),
        MaskCutout(threshold=0.5),
        AlgorithmicDenoise(mode='bm4d'),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    return prep1, prep2, prep4, prep5