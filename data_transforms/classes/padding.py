import torchio as tio
from utils.constants import TypeTripletInt


class PadDimTo(tio.SpatialTransform):
    def __init__(self, size, padding_mode: str|float = 0):
        super(PadDimTo, self).__init__()
        self.size = size
        self.padding = None
        self.padding_mode = padding_mode

    def apply_transform(self, subject):
        subject.check_consistent_space()

        spatial_shape = subject.spatial_shape
        padding_x = max(0, (self.size[0] - spatial_shape[0]) // 2)
        padding_y = max(0, (self.size[1] - spatial_shape[1]) // 2)
        padding_z = max(0, (self.size[2] - spatial_shape[2]) // 2)

        padding_x_extra = max(0, self.size[0] - spatial_shape[0] - 2 * padding_x)
        padding_y_extra = max(0, self.size[1] - spatial_shape[1] - 2 * padding_y)
        padding_z_extra = max(0, self.size[2] - spatial_shape[2] - 2 * padding_z)

        self.padding = (
            padding_x, padding_x + padding_x_extra,
            padding_y, padding_y + padding_y_extra,
            padding_z, padding_z + padding_z_extra
        )
        padded_data = tio.Pad(padding=self.padding, padding_mode=self.padding_mode)
        subject = padded_data(subject)

        return subject
    
    def inverse(self):
        return tio.Crop(self.padding)


class PadToRatio(tio.SpatialTransform):
    def __init__(self, ratio: TypeTripletInt, padding_mode: str|float = 0):
        super().__init__()
        self.ratio = ratio
        self.padding = None
        self.padding_mode = padding_mode

    def calculate_target_size(self, shape):
        ratio = self.ratio
        scale_factors = [shape[i] / r for i, r in enumerate(ratio)]
        max_scale = max(scale_factors)
        target_size = [int(r * max_scale) for r in ratio]
        return target_size

    def apply_transform(self, subject):
        subject.check_consistent_space()
        spatial_shape = subject.spatial_shape
        target_size = self.calculate_target_size(spatial_shape)
   
        self.padding = PadDimTo(size=target_size, padding_mode=self.padding_mode)
        subject = self.padding(subject)
        return subject

    def inverse(self):
        return self.padding.inverse()