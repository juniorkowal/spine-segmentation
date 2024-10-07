from monai.networks.nets import UNet
import torch.nn as nn
import torch


class MonaiUNet(nn.Module):
    def __init__(self, class_num=1, in_channels=1):
        super(MonaiUNet, self).__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=class_num,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    def forward(self, x):
        logits = self.unet(x)
        return logits


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(ConvBlock, self).__init__()

        if transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

        self.norm = nn.InstanceNorm3d(out_channels, affine=False)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class _UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(_UNet, self).__init__()

        features = init_features
        self.encoder1 = ConvBlock(in_channels, features, stride=2)
        self.encoder2 = ConvBlock(features, features*2, stride=2)
        self.encoder3 = ConvBlock(features*2, features*4, stride=2)
        self.encoder4 = ConvBlock(features*4, features*8, stride=2)

        self.middle = ConvBlock(features*8, features*16)

        self.decoder4 = ConvBlock(features*24, features*4, transpose=True, stride=2)
        self.decoder3 = ConvBlock(features*8, features*2, transpose=True, stride=2)
        self.decoder2 = ConvBlock(features*4, features, transpose=True, stride=2)

        self.final = nn.ConvTranspose3d(features*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        m = self.middle(e4)

        d4 = self.decoder4(torch.cat([m, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        output = self.final(torch.cat([d2, e1], dim=1))

        return output


if __name__ == "__main__":
    model = _UNet()
    # print(model)
    from torchinfo import summary
    summary(model, input_size=(1,1,96,96,128))
