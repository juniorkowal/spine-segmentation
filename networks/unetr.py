from monai.networks.nets.unetr import UNETR
import torch.nn as nn


class MonaiUNetR(nn.Module):
    def __init__(self, input_size=(96,96,128)):
        super(MonaiUNetR, self).__init__()
        self.model = UNETR(img_size=input_size, in_channels=1,out_channels=1,spatial_dims=3)

    def forward(self, x):
        logits = self.model(x)
        return logits
    

if __name__ == "__main__":
    from torchinfo import summary
    model = MonaiUNetR((96,96,128))
    # print(model)
    summary(model, input_size=(1,1,96,96,128))