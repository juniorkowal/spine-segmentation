from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn as nn


class MonaiSwinUNetR(nn.Module):
    def __init__(self, input_size=(96,96,128)):
        super(MonaiSwinUNetR, self).__init__()
        self.model = SwinUNETR(img_size=input_size, 
                               in_channels=1,
                               out_channels=1,
                               use_checkpoint=True,
                               spatial_dims=3,
                               )

    def forward(self, x):
        logits = self.model(x)
        return logits
    

if __name__ == "__main__":
    from torchinfo import summary
    shape = (96,96,128)
    model = MonaiSwinUNetR(shape)
    # print(model)
    summary(model, input_size=(1,1,*shape))