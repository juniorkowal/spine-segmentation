import torch
import torch.nn as nn

from .attention_unet_3d import AttentionUNet3D

'''Ablation functions for creating variations of the AttentionUNet3D
 model to isolate the impact of different components.
'''

# SCA3D ablation
#########################################################################################################
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

def change_sca3d_pool(in_channels=1, out_channels=1, pooling=nn.AdaptiveAvgPool3d, **kwargs):
    """Changes the pooling function inside channel_excitation for SCA3D."""
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           pool=pooling,
                           **kwargs)
#########################################################################################################


# Regular model ablation
#########################################################################################################
def depth_ablation(in_channels = 1, out_channels = 1, depth=6, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           depth=depth,
                           **kwargs)

def features_ablation(in_channels = 1, out_channels = 1, init_features=16, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           f_maps=init_features,
                           **kwargs)

def num_groups_ablation(in_channels = 1, out_channels = 1, num_groups=8, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           num_groups=num_groups,
                           **kwargs)

def layer_order_ablation(in_channels = 1, out_channels = 1, layer_order='crg', **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           layer_order=layer_order,
                           **kwargs)

def double_conv_ablation(in_channels = 1, out_channels = 1, double_conv=True, **kwargs):
    return AttentionUNet3D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           double_conv=double_conv,
                           **kwargs)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ablation_list = [
                 get_base_model(), no_channel_attention(), no_spatial_attention(), no_sca3d(),
                #  change_reduction_size(reduction=2), 
                 change_reduction_size(reduction=4),
                 change_reduction_size(reduction=16), change_reduction_size(reduction=32),
                 change_sca3d_act(act=nn.LeakyReLU), change_sca3d_act(act=nn.ELU), change_sca3d_act(act=Swish),
                 change_sca3d_pool(pooling=nn.AdaptiveMaxPool3d),
                 depth_ablation(depth=3), depth_ablation(depth=4), depth_ablation(depth=5),
                 features_ablation(init_features=8), features_ablation(init_features=4),
                #  num_groups_ablation(num_groups=4), num_groups_ablation(num_groups=16),
                 layer_order_ablation(layer_order='cr'), layer_order_ablation(layer_order='cl'),
                 layer_order_ablation(layer_order='ce'), layer_order_ablation(layer_order='cri'),
                 double_conv_ablation(double_conv=False)
                 ]


if __name__ == "__main__":
    from torchinfo import summary
    import gc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.randn(1, 1, image_size, image_size, image_size).to(device)

    # print("x size: {}".format(x.size()))
    
    # model = depth_ablation(in_channels=1, out_channels=1, depth=7
    #                          )
    with torch.no_grad():
        for i, model in enumerate(ablation_list):
            print(f"Model {i}")
            model = model.to(device)
            shape = (128,128,128)
            summary(model, input_size=(1,1,*shape)) # 16,608,918
            # print(model)
            out = model(x)
            print(f"{i}, out size: {out.size()}")
            del model
            del out
            gc.collect()
            torch.cuda.empty_cache() 