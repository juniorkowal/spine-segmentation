import os
import re
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import hdf5plugin
import numpy as np
import requests
import torch
import torchio as tio
from data_transforms.classes import TioSubjectNoWarnings
from tqdm import tqdm
from verse_utils.data_utilities import load_centroids


class VerSe(tio.SubjectsDataset):
    """VerSe dataset.

    Args:
        root (str or Path): Root directory of the dataset.
        split (str): Which split to use, must be one of 'training', 'validation', or 'test'.
        edition (str or int): Edition of VerSe dataset: 19 (VerSe19) or 20 (VerSe20).
        transform (callable, optional): An instance of :class:`~torchio.transforms.transform.Transform`.
        download (bool, optional): If set to ``True``, will download the data into :attr:`root`.

    Example:
        >>> import torchio as tio
        >>> transforms = [
        ...     tio.ToCanonical(),  # to RAS
        ...     tio.Resample((1, 1, 1)),  # to 1 mm iso
        ... ]
        >>> dataset = VerSe20(
        ...     'path/to/verse_root/',
        ...     split='training',
        ...     edition = '19',
        ...     transform=tio.Compose(transforms),
        ...     download=True,
        ... )
        >>> print('Number of subjects:', len(dataset))
        >>> sample_subject = dataset[0]
        >>> print('Keys in subject:', tuple(sample_subject.keys()))
    """

    base_urls = {
        '19': {
            'training': 'https://files.de-1.osf.io/v1/resources/jtfa5/providers/osfstorage/5ffca086e80d370320a594cc/?zip=',
            'validation': 'https://files.de-1.osf.io/v1/resources/jtfa5/providers/osfstorage/5ffcb524ba0109031c8931fa/?zip=',
            'test': 'https://files.de-1.osf.io/v1/resources/jtfa5/providers/osfstorage/5ffcbd71e80d370336a56f8c/?zip=',
        },
        '20': {
            'training': 'https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa463786541a01e714d390/?zip=',
            'validation': 'https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa463686541a01eb15048c/?zip=',
            'test': 'https://files.de-1.osf.io/v1/resources/4skx2/providers/osfstorage/5ffa4635ba010901f0891bd0/?zip=',
        }
    }

    alternative_urls = {
        '19': {
            'training': 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip',
            'validation': 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip',
            'test': 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip',
        },
        '20': {
            'training': 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip',
            'validation': 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip',
            'test': 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip',
        }
    }

    splits = ('training', 'validation', 'test')
    file_extensions = {
        'image': 'nii.gz',
        'label': 'nii.gz',
        'ctds': 'json',
    }
    exclude_pattern = 'msk'
    include_pattern = 'ctd'

    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'training',
        edition: Union[str, int] = '20',
        transform: Optional[tio.Transform] = None,
        download: bool = False,
        **kwargs,
    ):
        if str(edition) not in {'19', '20'}:
            raise ValueError("Edition must be '19' or '20'")
        self.edition = str(edition)

        self.root = Path(root) / f'verse{self.edition}'
        if split not in self.splits:
            raise ValueError(f'Split must be one of {self.splits}')
        self.split = split
        self.split_dir = self.root / f'dataset-0{self.splits.index(self.split) + 1}{self.split}'
        self.subjects = self._get_subjects(download)
        super().__init__(self.subjects, transform=transform, **kwargs)

    def _get_subjects(self, download: bool) -> Sequence[TioSubjectNoWarnings]:
        paths_dict = self._load_paths()

        if self._has_empty_values(paths_dict):
            if download:
                print(f"Files not found for {self.split} split of VerSe{self.edition}.\nDownloading files.")
                self._download()
                paths_dict = self._load_paths()
            else:
                raise RuntimeError(f'Dataset not found for {self.split} split of VerSe{self.edition}. You can use download=True to download it')

        subjects = []
        for image_path, label_path, ctds_path in zip(paths_dict['image'], paths_dict['label'], paths_dict['ctds']):
            subject_id = self.get_subject_id(image_path)
            subject = TioSubjectNoWarnings(
                subject_id=subject_id,
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                ctds=load_centroids(ctds_path),
            )
            # if subject_id == 'verse006':
            #     continue
            subjects.append(subject)

        return subjects

    def get_subject_id(self, sample_path: Path) -> str:
        filename = os.path.basename(sample_path)
        matches = re.findall(r'verse\d+', filename)
        subject_id = '_'.join(matches)
        return subject_id

    def _load_paths(self) -> dict:
        return {
            'image': self._get_paths(self.file_extensions['image'], exclude=self.exclude_pattern),
            'label': self._get_paths(self.file_extensions['label'], include=self.exclude_pattern, exclude='subreg'), # exclude subreg because there is some bad .nii.gz with that string in name
            'ctds': self._get_paths(self.file_extensions['ctds'], include=self.include_pattern),
        }

    def _get_paths(self, file_extension: str, include: Optional[str] = None, exclude: Optional[str] = None) -> list:
        include_pattern = f'*{include}*.{file_extension}' if include else f'*.{file_extension}'
        paths = (p for p in self.split_dir.rglob(include_pattern) if not p.name.startswith('.')
                 and (exclude is None or exclude not in p.name))
        sorted_paths = sorted(paths, key=lambda x: x.name)
        return sorted_paths

    def _has_empty_values(self, paths_dict: dict) -> bool:
        return any(not value for value in paths_dict.values())

    def _download(self):
        url = self.base_urls[self.edition][self.split]
        filename = f'dataset-verse{self.edition}{self.split}.zip'
        extract_dir = self.split_dir
        print(f"Downloading {filename}...")
        try:
            self._download_and_extract(url, filename, extract_dir)
        except Exception as e:
            print(f"Failed to download from {url}. Error: {e}")
            print("Trying alternative links...")
            url = self.alternative_urls[self.edition][self.split]
            print(f"Downloading from alternative link: {url}")
            self._download_and_extract(url, filename, extract_dir)

    @staticmethod
    def _download_and_extract(url: str, filename: str, extract_dir: Path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(filename, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        os.remove(filename)


class h5VerSe(tio.SubjectsDataset):
    """h5VerSe dataset for handling HDF5 data.

    Args:
        root (str or Path): Path to the HDF5 file containing the dataset.
        split (str): Which split to use, must be one of 'training', 'validation', or 'test'.
        transform (callable, optional): An instance of :class:`~torchio.transforms.transform.Transform`.
        patches (tuple, optional): Shape of patches to extract from the volumes.

    Example:
        >>> import torchio as tio
        >>> transform = tio.Compose([
        ...     tio.RandomAffine(),
        ...     tio.RandomElasticDeformation(),
        ... ])
        >>> dataset = h5VerSe(
        ...     'path/to/dataset.h5',
        ...     split='training',
        ...     transform=transform,
        ...     patches=(64, 64, 64),
        ... )
        >>> print('Number of subjects:', len(dataset))
        >>> sample_subject = dataset[0]
        >>> print('Keys in subject:', tuple(sample_subject.keys()))
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'training',
        transform: Optional[tio.Transform] = None,
        patches: Optional[tuple] = None,
        **kwargs,
    ):
        base_image_data = torch.zeros((1, 1, 1, 1))  # Dummy image to initialize SubjectsDataset class

        base_subject = tio.Subject(
            image=tio.ScalarImage(tensor=base_image_data),
            label=tio.LabelMap(tensor=base_image_data),
        )

        super().__init__(subjects=[base_subject])

        self.h5_path = root
        self.split = split
        self.transform = transform
        self.patches = patches

        self.h5_file = None
        self._sample_keys = None

    def _open_h5_file(self):
        if self.h5_file is None:
            if not os.path.exists(self.h5_path):
                raise FileNotFoundError(f"File not found: {self.h5_path}")
            self.h5_file = h5py.File(self.h5_path, 'r')

    def _close_h5_file(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except (ValueError, TypeError):
                # Handle the case where self.h5_file is already closed or destroyed
                pass
            finally:
                self.h5_file = None

    def _load_subject(self, sample_key: str) -> tio.Subject:
        self._open_h5_file()

        image_data = self.h5_file[self.split][sample_key][tio.IMAGE]
        labels_data = self.h5_file[self.split][sample_key][tio.LABEL]

        img_shape = image_data.shape[-3:]
        msk_shape = labels_data.shape[-3:]

        eq_shape = tuple(min(a, b) for a, b in zip(img_shape, msk_shape))

        if self.patches:
            lower_slice_range = tuple(max(0, (a - b)) for a, b in zip(eq_shape, self.patches))
            lower_slice_range = [(np.random.randint(low=0, high=a) if a != 0 else 0) for a in lower_slice_range]
            upper_slice_range = tuple(min(a + c, b) for a, b, c in zip(lower_slice_range, eq_shape, self.patches))
            shape = tuple(slice(a, b) for a, b in zip(lower_slice_range, upper_slice_range))
        else:
            shape = tuple(slice(0, a) for a in eq_shape)

        shape_img = (slice(0, image_data.shape[0]),) + shape if len(image_data.shape) > 3 else shape
        shape_msk = (slice(0, labels_data.shape[0]),) + shape if len(labels_data.shape) > 3 else shape

        image_data = image_data[shape_img].astype(np.float32)
        labels_data = labels_data[shape_msk]

        if labels_data.dtype != np.uint8:
            labels_data = labels_data.astype(np.float32)

        image_data = image_data[np.newaxis, ...] if len(image_data.shape) < 4 else image_data
        labels_data = labels_data[np.newaxis, ...] if len(labels_data.shape) < 4 else labels_data

        subject = tio.Subject(
            subject_id=sample_key,
            image=tio.ScalarImage(tensor=image_data),
            label=tio.LabelMap(tensor=labels_data),
        )

        if self.transform:
            subject = self.transform(subject)

        return subject

    def __len__(self) -> int:
        self._open_h5_file()
        length = len(self.h5_file[self.split])
        self._close_h5_file()
        return length

    def _load_keys(self):
        if self._sample_keys is None:
            self._sample_keys = list(self.h5_file[self.split].keys())        

    def __getitem__(self, index: int) -> tio.Subject:
        self._open_h5_file()

        self._load_keys()

        sample_key = self._sample_keys[index]
        subject = self._load_subject(sample_key)

        self._close_h5_file()

        return subject

    def __del__(self):
        self._close_h5_file()

    def get_subject_by_id(self, id: str) -> tio.Subject:
        self._open_h5_file()

        self._load_keys()

        if id in self._sample_keys:
            subject = self._load_subject(id)
            self._close_h5_file()
            return subject
        else:
            self._close_h5_file()
            raise ValueError(f"Sample with id '{id}' not found in the dataset.")



if __name__ == '__main__':
    ...
