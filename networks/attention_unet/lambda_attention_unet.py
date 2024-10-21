import time

import torch
import torch.nn as nn
from .attention_unet_3d import AttentionUNet3D
from .building_blocks import DoubleConv, SingleConv
from .lambda_layer import LambdaLayer3d



def replace_conv_layers(submodule, 
                        action="replace_single", 
                        delete_norm = False, 
                        lambda_config = {'global_context_size': None,
                                         'local_context_size': 5,
                                         'kdim': 16,
                                         'udim': 1,
                                         'num_heads': 4,
                                         'implementation': 1}):
    """
    Function to modify a submodule containing SingleConv and DoubleConv layers.
    Replaces convolution to lambdalayers.

    Parameters:
    - submodule: The PyTorch submodule to be modified.
    - action: The action to be performed. Options are:
        - "replace_double": Replace two conv blocks in DoubleConv with a single LambdaLayer3d.
                            Example: before: conv relu group norm, conv relu grop norm
                                     after: lambdalayer relu group norm
        - "replace_single": Replace the conv in SingleConv blocks with LambdaLayer3d.
                            Example: before: conv relu group norm, conv relu grop norm
                                     after: lambdalayer relu group norm, lambdalayer relu group norm
    - delete_norm: Whether to delete activation and normalization layer from block.
    """
    for name, module in submodule.named_children():
        if isinstance(module, DoubleConv):
            if action == "replace_double":
                in_channels = module.SingleConv1.conv.in_channels
                out_channels = module.SingleConv2.conv.out_channels

                del module.SingleConv1
                module.SingleConv2.conv = LambdaLayer3d(in_channels=in_channels, out_channels=out_channels, **lambda_config)

                if delete_norm:
                    del module.SingleConv2.ReLU
                    del module.SingleConv2.groupnorm

            elif action == "replace_single":
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, SingleConv):
                        in_channels = sub_module.conv.in_channels
                        out_channels = sub_module.conv.out_channels

                        sub_module.conv = LambdaLayer3d(in_channels=in_channels, out_channels=out_channels, **lambda_config)

                        if delete_norm:
                            del sub_module.ReLU
                            del sub_module.groupnorm
                     
    # print(f"Completed action '{action}' on {submodule}.")


def get_lambda_att_depth_5(action = 'replace_double', lambda_config = {}):
    model = AttentionUNet3D(in_channels=1, out_channels=1)
    for encoder in model.encoders[1:]:
        replace_conv_layers(encoder, action = action, lambda_config=lambda_config)
    for decoder in model.decoders[:4]:
        replace_conv_layers(decoder, action = action, lambda_config=lambda_config)
    return model


def get_lambda_att_depth_4(action = 'replace_double', lambda_config = {}):
    model = AttentionUNet3D(in_channels=1, out_channels=1)
    for encoder in model.encoders[2:]:
        replace_conv_layers(encoder, action = action, lambda_config=lambda_config)
    for decoder in model.decoders[:3]:
        replace_conv_layers(decoder, action = action, lambda_config=lambda_config)
    return model


def get_lambda_att_depth_3(action = 'replace_double', lambda_config = {}):
    model = AttentionUNet3D(in_channels=1, out_channels=1)
    for encoder in model.encoders[3:]:
        replace_conv_layers(encoder, action = action, lambda_config=lambda_config)
    for decoder in model.decoders[:2]:
        replace_conv_layers(decoder, action = action, lambda_config=lambda_config)
    return model


def get_lambda_att_depth_2(action = 'replace_double', lambda_config = {}):
    model = AttentionUNet3D(in_channels=1, out_channels=1)
    for encoder in model.encoders[4:]:
        replace_conv_layers(encoder, action = action, lambda_config=lambda_config)
    for decoder in model.decoders[:1]:
        replace_conv_layers(decoder, action = action, lambda_config=lambda_config)
    return model


def get_lambda_att_depth_1(action = 'replace_double', lambda_config = {}):
    model = AttentionUNet3D(in_channels=1, out_channels=1)
    replace_conv_layers(model.encoders[-1], action = action, lambda_config=lambda_config)
    return model


# hybrid model study
def get_hybrid_models(action = 'replace_double', 
                      lambda_config = {'global_context_size': None,
                                        'local_context_size': 5,
                                        'kdim': 32,
                                        'udim': 1,
                                        'num_heads': 4,
                                        'implementation': 1}):
    
    lambda_models = [get_lambda_att_depth_5(action, lambda_config),
                     get_lambda_att_depth_4(action, lambda_config),
                     get_lambda_att_depth_3(action, lambda_config),
                     get_lambda_att_depth_2(action, lambda_config),
                     get_lambda_att_depth_1(action, lambda_config)]
    return lambda_models


# Varying query depth, number of heads and intra-depth.
def hyperparameter_ablation():
    configs = [
        {'kdim': 8, 'num_heads': 2, 'udim': 1},
        {'kdim': 8, 'num_heads': 16, 'udim': 1},
        {'kdim': 2, 'num_heads': 4, 'udim': 1},
        {'kdim': 4, 'num_heads': 4, 'udim': 1},
        {'kdim': 8, 'num_heads': 4, 'udim': 1},
        {'kdim': 16, 'num_heads': 4, 'udim': 1},
        {'kdim': 32, 'num_heads': 4, 'udim': 1},
        {'kdim': 2, 'num_heads': 8, 'udim': 1},
        {'kdim': 4, 'num_heads': 8, 'udim': 1},
        {'kdim': 8, 'num_heads': 8, 'udim': 1},
        {'kdim': 16, 'num_heads': 8, 'udim': 1},
        {'kdim': 32, 'num_heads': 8, 'udim': 1},
        {'kdim': 8, 'num_heads': 8, 'udim': 4},
        {'kdim': 8, 'num_heads': 8, 'udim': 8},
        {'kdim': 16, 'num_heads': 4, 'udim': 4},
    ]
    return configs

# Content vs position interactions
def content_position_ablation():
    configs = [{'content': False},
               {'position': False}
               ]
    return configs

# Importance of scope size
def scope_size_ablation():
    configs = [{'local_context_size': 3},
               {'local_context_size': 7},
               {'local_context_size': 15},
               {'local_context_size': 23},
               {'local_context_size': 31},
            #    {'global_context_size': 4},
               ]
    return configs    


@torch.no_grad()
def time_model_forward_pass(get_lambda_models, data_in, device='cuda'):
    """
    Function to time each model's forward pass with the given input.
    
    Parameters:
    - get_lambda_models: Function to generate models (e.g., get_lambda_models())
    - data_in: The input tensor to be passed to the models
    - device: The device to perform computation on ('cuda' for GPU, 'cpu' for CPU)
    """
    
    for i, model in enumerate(get_lambda_models):
        model = model.to(device)
        data_in = data_in.to(device)

        if device == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        output = model(data_in)

        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
        else:
            elapsed_time = (time.time() - start_time) * 1000

        print(f"Model {i + 1}: Output shape: {output.shape}, Time taken: {elapsed_time:.2f} ms")
        # summary(model=model, input_data=data_in)


if __name__ == "__main__":
    import gc
    import time

    import torch
    from torchinfo import summary
    device = 'cuda'
    data_in = torch.rand(1,1,128,128,128)
    # model = AttentionUNet3D(in_channels=1, out_channels=1)

    # replace_conv_layers(model.encoders[4], action = 'replace_double')
    # replace_conv_layers(model.encoders[5], action = 'replace_double')

    # replace_conv_layers(model.decoders[0], action = 'replace_double')
    # replace_conv_layers(model.decoders[1], action = 'replace_double')
    # print(len(model.encoders))
    # print(len(model.decoders))
    # exit()

    # for i, encoder in enumerate(model.encoders[1:]):
    #     replace_conv_layers(encoder, action = 'replace_double')
    #     print(i)
    # for decoder in model.decoders[:4]:
    #     replace_conv_layers(decoder, action = 'replace_double')

    # exit()

    # for model in get_lambda_models():
    #     model = model.to(device)
    #     data_in = data_in.to(device)

    #     print(model(data_in).shape)
    #     summary(model = model, input_data=data_in)




    lambda_configs = scope_size_ablation()

    for i, config in enumerate(lambda_configs):
        time_model_forward_pass([get_hybrid_models(lambda_config=config)[-1]], data_in, device='cuda')
