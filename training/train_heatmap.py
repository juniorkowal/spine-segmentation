import pytest
pytest.main([__file__])

import os

import torch.optim as optim
import torchio as tio
from data_transforms.heatmap_transforms import (get_heatmap_postpro,
                                                get_heatmap_prepro,
                                                heatmap_transforms)
from dataloader import h5VerSe
from metrics.losses import *
from metrics.metric_handler import MetricHandler
from networks import *
from prodigyopt import Prodigy
from torch.utils.data import DataLoader
from training.scripts import (SET_SEED, get_args, load_model, train_loop,
                              validate, get_model)
from utils.constants import DEVICE, HEATMAP, MODELS_PATH, NUM_WORKERS
from utils.visualize import visualize_spine_localization_heatmap_detailed



def train_heatmap(args):
    model = get_model(args)
    
    spine_loc_metrics = MetricHandler()
    spine_loc_metrics.add_metric(name='MSE', metric_fn=MSE(), is_accuracy=False) # loss
    
    eval_metrics = { # acc
        'bbIoU': bbIoU(threshold=0.5), 
        'RMSE': RMSE(), 
        'MAE': MAE(), 
        'R2': R2()
    }
    
    for name, fn in eval_metrics.items():
        spine_loc_metrics.add_metric(name=name, metric_fn=fn.to(DEVICE), is_accuracy=True)
    
    optimizer = Prodigy(params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    config = {
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'input_shape': args.input_shape,
        'dataset_path': args.data_path,
        'dataset_edition': args.dataset_edition,
        'early_stopping': args.early_stopping,
        'target': HEATMAP,
        'model_name': args.model_name,
        'run_id': args.run_id
    }
    
    transform = heatmap_transforms(args.input_shape)
    train_loop(config=config,
               metric_handler=spine_loc_metrics,
               model=model,
               model_name=args.model_name,
               transforms_func=transform,
               visualize_func=visualize_spine_localization_heatmap_detailed,
               optimizer=optimizer,
               scheduler=scheduler,
               use_wandb=args.use_wandb)

    test_transform = transform[2]
    test_dataset = h5VerSe(root=config['dataset_path'], split='test', transform=test_transform,
                           edition=config['dataset_edition'], download=True, patches=config['input_shape'])
    
    best_model = next((f for f in os.listdir(os.path.join(MODELS_PATH, config['model_name'])) if 'BEST' in f), None)
    
    if best_model:
        print(f'\nRunning model through test set: {best_model}\n')
        load_model(model, optimizer, scheduler, model_path=os.path.join(MODELS_PATH, config['model_name'], best_model))
        loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=NUM_WORKERS)
        
        for target, file_suffix in [(HEATMAP, 'test_set_result.txt'), (tio.LABEL, 'test_set_result_label_bbiou.txt')]:
            best_result = validate(model=model, dataloader=loader, metric_handler=spine_loc_metrics, target=target)
            result_path = os.path.join('trained_models', config['model_name'], file_suffix)
            with open(result_path, 'w') as f:
                f.writelines(str(best_result))



if __name__ == "__main__":
    SET_SEED(42)
    args = get_args()
    train_heatmap(args)
    