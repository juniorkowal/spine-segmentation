import pprint
import warnings
from typing import Optional

import numpy as np
import torchio as tio
from utils.constants import TypeSpatialShape


class TioSubjectNoWarnings(tio.Subject):

    def check_consistent_attribute(
        self,
        attribute: str,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-6,
        message: Optional[str] = None,
    ) -> None:

        message = (
            f'More than one value for "{attribute}" found in subject images:\n{{}}'
        )

        names_images = self.get_images_dict(intensity_only=False).items()
        try:
            first_attribute = None
            first_image = None

            for image_name, image in names_images:
                if first_attribute is None:
                    first_attribute = getattr(image, attribute)
                    first_image = image_name
                    continue
                current_attribute = getattr(image, attribute)
                all_close = np.allclose(
                    current_attribute,
                    first_attribute,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                if not all_close:
                    message = message.format(
                        pprint.pformat(
                            {
                                first_image: first_attribute,
                                image_name: current_attribute,
                            }
                        ),
                    )
                    # print(message)
        except TypeError:
            # fallback for non-numeric values
            values_dict = {}
            for image_name, image in names_images:
                values_dict[image_name] = getattr(image, attribute)
            num_unique_values = len(set(values_dict.values()))
            if num_unique_values > 1:
                message = message.format(pprint.pformat(values_dict))
                raise RuntimeError(message) from None


class ConsistentSpaceFix(tio.SpatialTransform):
    def __init__(self, enforce_size = None):
        super(ConsistentSpaceFix, self).__init__()
        self.enforce_size = enforce_size

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        imgs = subject.get_images_names()
        if self.enforce_size is not None:
            spatial_shape = self.enforce_size
        else:
            spatial_shape = subject[tio.IMAGE][tio.DATA].shape[-3:]
        crop_pad = tio.CropOrPad(target_shape=spatial_shape)

        for i in range(len(imgs)):
            subject[imgs[i]] = crop_pad(subject[imgs[i]])
        return subject


class TioResize(tio.SpatialTransform):
    """Resample images so the output shape matches the given target shape.

    The field of view remains the same.

    .. warning:: In most medical image applications, this transform should not
        be used as it will deform the physical object by scaling anistropically
        along the different dimensions. The solution to change an image size is
        typically applying :class:`~torchio.transforms.Resample` and
        :class:`~torchio.transforms.CropOrPad`.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`. The size of dimensions set to
            -1 will be kept.
        image_interpolation: See :ref:`Interpolation`.
        label_interpolation: See :ref:`Interpolation`.
    """

    def __init__(
        self,
        target_shape: TypeSpatialShape,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = np.asarray(tio.utils.to_tuple(target_shape, length=3))
        self.image_interpolation = self.parse_interpolation(
            image_interpolation,
        )
        self.label_interpolation = self.parse_interpolation(
            label_interpolation,
        )
        self.args_names = [
            'target_shape',
            'image_interpolation',
            'label_interpolation',
        ]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        shape_in = np.asarray(subject.spatial_shape)
        shape_out = self.target_shape
        negative_mask = shape_out == -1
        shape_out[negative_mask] = shape_in[negative_mask]
        spacing_in = np.asarray(subject.spacing)
        spacing_out = shape_in / shape_out * spacing_in
        resample = tio.Resample(
            spacing_out,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
        )
        resampled = resample(subject)
        assert isinstance(resampled, tio.Subject)
        # Sometimes, the output shape is one voxel too large
        # Probably because Resample uses np.ceil to compute the shape
        if any(resampled[key].shape[-3:] != tuple(shape_out) for key in [tio.IMAGE, tio.LABEL]):
            message = (
                f'Output shape != target shape {tuple(shape_out)}. Fixing with CropOrPad'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            crop_pad = ConsistentSpaceFix(shape_out)  # type: ignore[arg-type]
            resampled = crop_pad(resampled)
        assert isinstance(resampled, tio.Subject)
        return resampled