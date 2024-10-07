import os

# from preprocessing import Preprocessing, basic_dataset_prepro
import torchio as tio
# import torch.nn as nn
# import wandb
# from data_transforms import heatmap_transforms
# from data_transforms.classes import (AlgorithmicDenoise, CubicResize,
#                                      LabelToHeatmap, PadDimTo, PadToRatio)
from dataloader import h5VerSe
from metrics.losses import *
from metrics.metric_handler import MetricHandler
from networks import *
from preprocessing import dataset_preprocessing
from prodigyopt import Prodigy
from torch.utils.data import DataLoader
# from torchinfo import summary
# from tqdm.autonotebook import tqdm
# from training.scripts import *
from training.scripts import load_model, train_loop, validate
from utils.constants import DEVICE, HEATMAP, MODELS_PATH, NUM_WORKERS
from utils.visualize import visualize_spine_localization_heatmap_detailed
from data_transforms.heatmap_transforms import get_heatmap_prepro, get_heatmap_postpro, heatmap_transforms



def train_heatmap(data_path, transform, model_name):

    spine_loc_metrics = MetricHandler()
    
    spine_loc_metrics.add_metric(name='MSE', metric_fn=MSE(), is_accuracy=False) # loss
    eval_metrics = {'bbIoU':bbIoU(threshold=0.5), # eval
                    'RMSE':RMSE(), 
                    'MAE':MAE(),
                    'R2': R2()}

    for name, fn in eval_metrics.items():
        spine_loc_metrics.add_metric(name=name, metric_fn=fn.to(DEVICE), is_accuracy=True)

    shape = (64,64,128)

    epochs = 200
    # epochs = 1

    lr = 1.

    model = MonaiUNet(in_channels=1, class_num = 1).to(DEVICE)

    optimizer = Prodigy(params=model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    config = {
        'learning_rate': lr, 
        'epochs': epochs,
        'batch_size': 1, 
        'input_shape': shape,
        'dataset_path': data_path,
        'dataset_edition': '19',
        'early_stopping':30,
        'target': HEATMAP,
    }


    train_loop(config=config, 
               metric_handler=spine_loc_metrics,
               model=model,
               model_name=model_name,
               transforms_func=transform,
               visualize_func=visualize_spine_localization_heatmap_detailed,
               optimizer=optimizer,
               scheduler=scheduler,
               use_wandb=False)

    # test thing
    test_transform = transform[2]
    test_dataset = h5VerSe(root=data_path, split='test', transform=test_transform, edition=config['dataset_edition'], download=True, patches=shape)
    
    model_folder = os.path.join(MODELS_PATH, model_name)
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
    best_model = None
    for m in model_files:
        if 'BEST' in m:
            best_model = m

    print('\nRunning model through test set.',best_model,'\n')
    load_model(model, optimizer, scheduler, model_path=os.path.join(MODELS_PATH, model_name, best_model))
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # CHECK MODEL METRICS ON TEST SET
    best_result = validate(model=model, dataloader=loader, metric_handler=spine_loc_metrics, target=HEATMAP)
    result_path = os.path.join('trained_models', model_name, 'test_set_result.txt')
    with open(result_path, mode='w') as f:
        f.writelines(str(best_result))

    # CHECK bboxIOU BETWEEN REGRESSED HEATMAP AND MASK
    best_result = validate(model=model, dataloader=loader, metric_handler=spine_loc_metrics, target=tio.LABEL)
    result_path = os.path.join('trained_models', model_name, 'test_set_result_label_bbiou.txt')
    with open(result_path, mode='w') as f:
        f.writelines(str(best_result))


if __name__ == "__main__":
    shape = (64,64,128)
    preps = get_heatmap_prepro(shape)
    postpro = get_heatmap_postpro(shape)
    basic_transforms = heatmap_transforms(shape)

    # preprocessing experiments # TURN THIS INTO FUNCTION FOR TESTING PREPRO
    for i, preprocessing in enumerate(preps):
        dataset_preprocessing(out_name=f"prep{i}",transform=preprocessing,edition=19)
        train_heatmap(data_path=f'datasets/prep{i}/prep{i}.h5', transform=basic_transforms, model_name=f"prep{i}")

    for i, postpro in enumerate(postpro):
        def transform_wrapper():
            return None, postpro[0], postpro[1], None, None
        train_heatmap(data_path=IDK SPECIFY THE PATH IN FUNCTION, transform=transform_wrapper, model_name=f"postpro{i}")
    