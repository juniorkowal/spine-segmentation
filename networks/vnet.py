from monai.networks.nets.vnet import VNet
import torch.nn as nn


class MonaiVNet(nn.Module):
    def __init__(self):
        super(MonaiVNet, self).__init__()
        self.model = VNet(in_channels=1, out_channels=1, spatial_dims=3)
        self.replace_batchnorm(self.model)

    def replace_batchnorm(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                setattr(module, name, nn.InstanceNorm3d(child.num_features))
            else:
                self.replace_batchnorm(child)

    def forward(self, x):
        logits = self.model(x)
        return logits
    

if __name__ == "__main__":
    from torchinfo import summary
    model =  MonaiVNet()
    # print(model)
    summary(model, input_size=(1,1,96,96,128))