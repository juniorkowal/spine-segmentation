import torch
import torch.nn as nn

from .attention_unet_3d import AttentionUNet3D

'''Ablation functions for creating variations of the AttentionUNet3D
 model to isolate the impact of different components.
'''

def get_base_model(in_channels = 1, out_channels = 1, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           **kwargs)

def no_channel_attention(in_channels = 1, out_channels = 1, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           channel_attention=False,
                           **kwargs)

def no_spatial_attention(in_channels = 1, out_channels = 1, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           spatial_attention=False,
                           **kwargs)

def no_sca3d(in_channels = 1, out_channels = 1, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           spatial_attention=False,
                           channel_attention=False,
                           **kwargs)

def change_reduction_size(in_channels = 1, out_channels = 1, reduction=8, **kwargs):
    '''Check influence of the hidden dim in SCA3D; default: 8'''
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           reduction=reduction,
                           **kwargs)

def change_sca3d_act(in_channels=1, out_channels=1, act=nn.ReLU, **kwargs):
    """Changes the activation function inside channel_excitation for SCA3D."""
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           activation=act,
                           **kwargs)



if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.randn(1, 1, image_size, image_size, image_size).to(device)

    print("x size: {}".format(x.size()))
    
    model = change_sca3d_act(in_channels=1, out_channels=1, act=nn.PReLU
                             )

    model = model.to(device)
    shape = (128,128,128)
    summary(model, input_size=(1,1,*shape)) # 16,608,918
    # print(model)
    out = model(x)
    print("out size: {}".format(out.size()))