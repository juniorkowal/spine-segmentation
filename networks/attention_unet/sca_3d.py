import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA3D(nn.Module):
    def __init__(self, in_channels, reduction=16, 
                 channel_attention=True, 
                 spatial_attention=True,
                 activation=nn.ReLU,
                 pool=nn.AdaptiveAvgPool3d,
                 **kwargs):
        super().__init__()

        if hasattr(activation, 'inplace'):
            activation_layer = activation(inplace=True)
        else:
            activation_layer = activation()
            
        if channel_attention:
            self.avg_pool = pool(1)
            self.channel_excitation = nn.Sequential(nn.Linear(in_channels, int(in_channels // reduction)),
                                                    activation_layer,#nn.ReLU(inplace=True),
                                                    nn.Linear(int(in_channels // reduction), in_channels))
        
        if spatial_attention:
            self.spatial_se = nn.Conv3d(in_channels, 1, kernel_size=1,
                                        stride=1, padding=0, bias=False)
        
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

    def forward(self, x):
        B, C, D, H, W = x.size()

        if self.channel_attention:
            chn_se = self.avg_pool(x).view(B, C)
            chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(B, C, 1, 1, 1))
            chn_se = torch.mul(x, chn_se)

        if self.spatial_attention:
            spa_se = torch.sigmoid(self.spatial_se(x))
            spa_se = torch.mul(x, spa_se)

        if self.channel_attention and self.spatial_attention:
            net_out = chn_se + x + spa_se
        elif self.channel_attention:
            net_out = chn_se + x
        elif self.spatial_attention:
            net_out = spa_se + x
        else:
            net_out = x

        return net_out
