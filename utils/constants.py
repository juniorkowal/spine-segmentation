import multiprocessing
from pathlib import Path
from typing import Tuple, Union

import h5py
import hdf5plugin
import torch
import torchio as tio
from verse_utils.data_utilities import load_centroids

remapping_cervical_thoraic_lumbar = {
    1: 1,  2: 1,  3: 1,  4: 1,  5: 1,  6: 1,  7: 1,  # Cervical spine (C1-C7)
    8: 2,  9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2,  # Thoracic spine (T1-T12)
    20: 3, 21: 3, 22: 3, 23: 3, 24: 3, 25: 3,  # Lumbar spine (L1-L6)
    26: 2,  # T13 (Thoracic spine)
    28: 2   # T13 (Thoracic spine)
}

remapping_binary = {key: 1 for key in list(range(1, 29))}


HEATMAP = 'heatmap'
CTDS = 'ctds'
COMPRESSION = hdf5plugin.Blosc2(cname = 'lz4', clevel = 5, filters = hdf5plugin.Blosc2.BITSHUFFLE)
MODELS_PATH = 'trained_models'
H5 = 'h5'
NUM_WORKERS = multiprocessing.cpu_count() # 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'


test_img_path = Path('verse_utils/test_images')
empty_tensor = torch.zeros((1, 1, 1, 1))

empty_image = tio.ScalarImage(tensor=empty_tensor)
empty_label = tio.LabelMap(tensor=empty_tensor)
empty_centroids = {}

try:
    TEST_IMAGE = tio.ScalarImage(test_img_path / 'sub-verse014_ct.nii.gz')
    TEST_LABEL = tio.LabelMap(test_img_path / 'sub-verse014_seg-vert_msk.nii.gz')
    TEST_CENTROIDS = load_centroids(test_img_path / 'sub-verse014_seg-vb_ctd.json')
except FileNotFoundError:
    TEST_IMAGE, TEST_LABEL, TEST_CENTROIDS = empty_image, empty_label, empty_centroids

TEST_SUBJECT = tio.Subject(image=TEST_IMAGE, label=TEST_LABEL, ctds=TEST_CENTROIDS, subject_id='test')


TypeTripletInt = Tuple[int, int, int]
TypeSpatialShape = Union[int, TypeTripletInt]
