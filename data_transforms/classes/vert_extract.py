import torch
import torchio as tio
from utils.misc import recalculate_centroids


class ExtractVertebrae(tio.SpatialTransform):
    def __init__(self, patch_size: tuple, only_centered_vert = False):
        super(ExtractVertebrae, self).__init__()
        self.patch_size = patch_size
        self.only_centered_vert = only_centered_vert

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in subject.get_images(intensity_only=False):
            if image[tio.TYPE] == tio.LABEL:
                label_map = image
                break

        shape = subject.spatial_shape
        ctds = recalculate_centroids(label_map.data.squeeze())
        patches_label = torch.zeros((len(ctds), *self.patch_size))
        patches_image = torch.zeros((len(ctds), *self.patch_size))

        for i, (label, x, y, z) in enumerate(ctds):
            x_start = int(x - self.patch_size[0] // 2)
            x_end = x_start + self.patch_size[0]
            y_start = int(y - self.patch_size[1] // 2)
            y_end = y_start + self.patch_size[1]
            z_start = int(z - self.patch_size[2] // 2)
            z_end = z_start + self.patch_size[2]

            padx_start = padx_end = pady_start = pady_end = padz_start = padz_end = 0

            if x_start < 0:
                padx_start = -x_start
                x_start = 0
            if x_end > shape[0]:
                padx_end = x_end - shape[0]
                x_end = shape[0]
            
            if y_start < 0:
                pady_start = -y_start
                y_start = 0
            if y_end > shape[1]:
                pady_end = y_end - shape[1]
                y_end = shape[1]

            if z_start < 0:
                padz_start = -z_start
                z_start = 0
            if z_end > shape[2]:
                padz_end = z_end - shape[2]
                z_end = shape[2]
                
            image_patch = subject[tio.IMAGE][tio.DATA][:, x_start:x_end, y_start:y_end, z_start:z_end]
            label_patch = subject[tio.LABEL][tio.DATA][:, x_start:x_end, y_start:y_end, z_start:z_end].clone()

            if self.only_centered_vert:
                label_patch[label_patch!=label] = 0

            if any((padx_start, padx_end, pady_start, pady_end, padz_start, padz_end)):
                padding = tio.Pad(padding=(padx_start, padx_end, pady_start, pady_end, padz_start, padz_end))
                image_patch = padding(image_patch)
                label_patch = padding(label_patch)

            assert image_patch.size()[-3:] == self.patch_size, f"Patch size should be {self.patch_size}, not {image_patch.size()[-3:]}"

            patches_label[i] = label_patch.data.squeeze()
            patches_image[i] = image_patch.data.squeeze()

        subject[tio.IMAGE].set_data(patches_image)
        subject[tio.LABEL].set_data(patches_label)
        return subject
