import warnings

import torch
import torch.nn.functional as F
import torchio as tio
from bm4d import BM4DProfile, bm4d
from utils.torch_nlm import nlm3d


class AlgorithmicDenoise(tio.IntensityTransform):
    def __init__(self, mode = 'nlm3d', device = 'cpu', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.device = device

        profile = BM4DProfile()
        profile.max_stack_size_wiener = 16
        profile.search_window_ht = (4, 4, 4)
        profile.search_window_wiener = profile.search_window_ht
        profile.split_block_extraction = [False, True, True]

        self.profile = profile

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in self.get_images(subject):
            assert isinstance(image, tio.ScalarImage)

            img_torch = image[tio.DATA]

            if img_torch.min() < 0 or img_torch.max() > 1:
                warnings.warn("Image intensity not in range [0, 1]. Rescaling intensity.")
                rescale = tio.RescaleIntensity(out_min_max=(0, 1))
                img_torch = rescale(img_torch)

            std = self._estimate_noise_std_3d(img_torch)
            if self.mode == 'bm4d':
                # raise NotImplementedError
                # print(img_torch.shape)
                img_np = bm4d(img_torch.squeeze().numpy(), sigma_psd = std, profile=self.profile)
                image.set_data(tensor = torch.from_numpy(img_np).unsqueeze(0))
            else:
                img_np = nlm3d(img_torch.to(self.device), std = std).to(self.device)
                image.set_data(tensor = img_np.cpu().unsqueeze(0))
                torch.cuda.empty_cache()

        return subject

    def _estimate_noise_std_3d(self, image):
        if image.dim() == 5 and image.shape[1] > 1:
            image = image.mean(dim=1, keepdim=True)

        filter_x = torch.tensor([[-1, 1]], dtype=image.dtype, device=image.device).view(1, 1, 1, 1, 2)
        filter_y = torch.tensor([[-1], [1]], dtype=image.dtype, device=image.device).view(1, 1, 1, 2, 1)
        filter_z = torch.tensor([[-1], [1]], dtype=image.dtype, device=image.device).view(1, 1, 2, 1, 1)

        padded_image_x = F.pad(image, (1, 0, 0, 0, 0, 0), mode='reflect')
        padded_image_y = F.pad(image, (0, 0, 1, 0, 0, 0), mode='reflect')
        padded_image_z = F.pad(image, (0, 0, 0, 0, 1, 0), mode='reflect')

        gradients_x = F.conv3d(padded_image_x, filter_x)
        gradients_y = F.conv3d(padded_image_y, filter_y)
        gradients_z = F.conv3d(padded_image_z, filter_z)

        gradients = torch.sqrt(gradients_x ** 2 + gradients_y ** 2 + gradients_z ** 2)
        mad = torch.median(torch.abs(gradients - torch.median(gradients)))
        noise_std = mad / 0.6745

        return noise_std.item()