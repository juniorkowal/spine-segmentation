import torch
import torchio as tio
from utils.constants import TEST_SUBJECT
from utils.misc import compute_bbox3d


class LabelToBbox(tio.Transform):
    r"""Computes and sets the 3D bounding box coordinates as a flat tensor list.

    This transformation computes the 3D bounding box coordinates from a label map using the
    `compute_bbox3d` function and sets these coordinates as the new label data in the subject.
    
    The resulting label will contain a flat tensor with the bounding box coordinates: 
    `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    
    Args:
        **kwargs: Additional keyword arguments for torchio.transforms.Transform.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        label = subject[tio.LABEL][tio.DATA].squeeze()
        bbox_min, bbox_max = compute_bbox3d(label)
        bbox_min, bbox_max = torch.tensor(bbox_min, dtype=torch.float32), torch.tensor(bbox_max, dtype=torch.float32)
        d, h, w = subject.spatial_shape
        
        bbox_min[0] /= d
        bbox_max[0] /= d
        bbox_min[1] /= h
        bbox_max[1] /= h
        bbox_min[2] /= w
        bbox_max[2] /= w        
        bbox_coords = torch.tensor(bbox_min.tolist() + bbox_max.tolist(), dtype=torch.float32)
        subject['bbox'] = {'data': bbox_coords}
        return subject
    


if __name__ == "__main__":
    transform = LabelToBbox()
    # TEST_SUBJECT.plot()
    bboxed = transform(TEST_SUBJECT)
    # bboxed.plot()
    print(bboxed['bbox'])
    print(TEST_SUBJECT[tio.LABEL])