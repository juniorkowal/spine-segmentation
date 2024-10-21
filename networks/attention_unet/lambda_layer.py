
import math
import timeit
from typing import Optional, Union

import einops
import torch
import torch.fft
import torch.nn as nn
import torch.nn.parallel
from torch.nn.modules.utils import _quadruple



class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:Union[int, tuple],
                 stride:Union[int, tuple] = (1, 1, 1, 1),
                 padding:Union[int, tuple] = (0, 0, 0, 0),
                 dilation:Union[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super(Conv4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out



class LambdaLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        global_context_size: Optional[int] = None,
        local_context_size: Optional[int] = None,
        kdim: int = 16,
        udim: int = 1,
        num_heads: int = 4,
        implementation: int = 1,
        content: bool = True,
        position: bool = True,
    ):
        """
        Lambda Networks module implemented for 5D input tensor (B, C, D, H, W).

        References:
            - [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://arxiv.org/abs/2102.08602)

        Args:
            in_channels: Input channel dimension.
            out_channels: Output channel dimension. Defaults to input dimension if not specified.
            global_context: Global Context size. If provided, the spatial dimensions (D, H, W) must match `m` exactly.
            local_context: Local context convolutional receptive field size (odd number).
            kdim: Key/Query dimension. Defaults to 16.
            udim: Intra-depth dimension for multi-query lambdas (corresponds to 'u' in the paper).
            num_heads: Number of heads in multi-query lambda layer (corresponds to 'h' in the paper).
            implementation: Integer flag representing which implementation should be utilized.

                Implementation 0: 
                    Implementation from the paper, constructing a n-D Lambda Module utilizing a (n+1)-D
                    Convolutional operator.

                Implementation 1: 
                    Equivalent implementation of the paper, constructing a n-D Lambda Module utilizing a
                    n-D Convolutional operator, and then looping through the Key (K) dimension, applying the
                    n-D conv to each K_i, finally concatenating all the values to map `u` -> `k`. 
                    Equivalent to Impl 0 for fp64, with a minor loss of floating point precision at fp32. 
                    May cause issues at fp16 (untested).
            
            content/position: Flags for ablation study. 
                    
        """
        super().__init__()
        self.content, self.position = content, position

        dim_out = out_channels if out_channels is not None else in_channels
        self.dim_in = in_channels
        self.dim_out = dim_out

        self.k = kdim
        self.u = udim
        self.h = num_heads
        self.m = global_context_size
        self.r = self.local_context = local_context_size

        if implementation not in (0, 1):
            raise ValueError("Implementation must be 0 or 1.")
        self.implementation = implementation

        if dim_out % num_heads != 0:
            raise ValueError("Output dimension must be divisible by the number of heads.")
        self.v = dim_out // num_heads

        self.to_q = nn.Conv3d(in_channels, kdim * num_heads, 1, bias=False)
        self.to_k = nn.Conv3d(in_channels, kdim * udim, 1, bias=False)
        self.to_v = nn.Conv3d(in_channels, self.v * udim, 1, bias=False)

        nn.init.normal_(self.to_q.weight, std=(kdim * dim_out) ** -0.5)
        nn.init.normal_(self.to_k.weight, std=dim_out ** -0.5)
        nn.init.normal_(self.to_v.weight, std=dim_out ** -0.5)

        self.norm_q = nn.BatchNorm3d(kdim * num_heads)# self.norm_q = nn.GroupNorm(num_groups=4, num_channels=dim_k * heads)
        self.norm_v = nn.BatchNorm3d(self.v * udim)# self.norm_v = nn.GroupNorm(num_groups=4, num_channels=self.v * dim_intra)
        
        if (global_context_size is None) == (local_context_size is None):
            raise ValueError("One of `m` (global) or `r` (local) context must be provided, but not both.")

        if self.local_context:
            assert (self.r % 2) == 1, "Receptive kernel size should be odd."
            padding = (local_context_size // 2,) * 3
            if self.implementation == 0:
                self.pos_conv = Conv4d(udim, kdim, (1, self.r, self.r, self.r), padding=(0, *padding))
            if self.implementation == 1:
                self.pos_conv = nn.Conv3d(udim, kdim, (self.r, self.r, self.r), padding=padding)
        else:
            assert self.m is not None, "You must specify the window size (m = h = w = d)"
            rel_lengths = 2 * self.m - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, rel_lengths, kdim, udim))
            self.rel_pos = self.compute_relative_positions(self.m, self.m, self.m)
            nn.init.uniform_(self.rel_pos_emb)

    def compute_relative_positions(self, d, h, w, device=None):
        pos = torch.meshgrid(torch.arange(d, device=device), torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        pos = einops.rearrange(torch.stack(pos), "n i j k -> (i j k) n")

        rel_pos = pos[None, :] - pos[:, None]
        rel_pos = torch.clamp(rel_pos, -self.m, self.m)
        rel_pos += self.m - 1
        return rel_pos
    
    def compute_local_context(self, q, v, dd, hh, ww):
        if self.implementation == 0:
            v = einops.rearrange(v, "b u v (dd hh ww) -> b u v dd hh ww", dd=dd, hh=hh, ww=ww)
            lambda_p = self.pos_conv(v)
        elif self.implementation == 1:
            v = einops.rearrange(v, "b u v (dd hh ww) -> b u v dd hh ww", dd=dd, hh=hh, ww=ww)
            v_stack = [self.pos_conv(v[:, :, v_idx, :, :, :]) for v_idx in range(self.v)]
            lambda_p = torch.stack(v_stack, dim=2)
        y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p.flatten(3))
        return y_p

    def compute_global_context(self, q, v, dd, hh, ww):
        if any(dim > self.m for dim in (hh, ww, dd)):
            raise ValueError(f"Input dimensions ({dd}, {hh}, {ww}) exceed global context size ({self.m}).")
        if (hh, ww, dd) != (self.m, self.m, self.m):
            self.rel_pos = self.compute_relative_positions(dd, hh, ww, device=q.device)
        d_, h_, w_ = self.rel_pos.unbind(dim=-1)
        rel_pos_emb = self.rel_pos_emb[d_, h_, w_]
        lambda_p = torch.einsum("n m k u, b u v m -> b n k v", rel_pos_emb, v)
        y_p = torch.einsum("b h k n, b n k v -> b h v n", q, lambda_p)
        return y_p

    def forward(self, x):
        b, c, dd, hh, ww = x.shape
        u = self.u
        h = self.h

        q = self.norm_q(self.to_q(x))
        v = self.norm_v(self.to_v(x))
        k = self.to_k(x)

        q = einops.rearrange(q, "b (h k) dd hh ww -> b h k (dd hh ww)", h=h)
        v = einops.rearrange(v, "b (u v) dd hh ww -> b u v (dd hh ww)", u=u)
        k = einops.rearrange(k, "b (u k) dd hh ww -> b u k (dd hh ww)", u=u)
        k = k.softmax(dim=-1)
        
        lambda_c = torch.einsum("b u k m, b u v m -> b k v", k, v)
        y_c = torch.einsum("b h k n, b k v -> b h v n", q, lambda_c)

        if self.local_context:
            y_p = self.compute_local_context(q, v, dd, hh, ww)
        else:
            y_p = self.compute_global_context(q, v, dd, hh, ww)

        # ablation
        #######################################################################################
        if self.content and self.position:
            Y = y_c + y_p
        elif self.content:
            Y = y_c
        elif self.position:
            Y = y_p
        #######################################################################################

        out = einops.rearrange(Y, "b h v (dd hh ww) -> b (h v) dd hh ww", dd=dd, hh=hh, ww=ww)
        return out

    def extra_repr(self):
        return f'input_dim={self.dim_in}, output_dim={self.dim_out}, m={self.m}, r={self.r}, k={self.k}, h={self.h}, u={self.u}'



if __name__ == "__main__":
    from torchinfo import summary

    device = 'cuda'
    data_in = torch.rand(1,32,16,16,16)
    lambda_layer = LambdaLayer3d(in_channels=32, 
                                 out_channels=16, 
                                 local_context_size=15, 
                                 global_context_size=None, 
                                 kdim=16, 
                                 num_heads=4, 
                                 implementation=1,
                                 )
    lambda_layer = lambda_layer.to(device)
    data_in = data_in.to(device)

    print(lambda_layer(data_in).shape)
    # summary(model = lambda_layer, input_data=data_in)
