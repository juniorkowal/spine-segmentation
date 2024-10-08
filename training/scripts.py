import argparse
import os
import random
import re
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import wandb
from dataloader import VerSe, h5VerSe
from metrics.metric_handler import MetricHandler
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.autonotebook import tqdm
from utils.constants import DEVICE, H5, MODELS_PATH, NUM_WORKERS
from networks import *

# from tqdm.notebook import tqdm



def SET_SEED(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

SET_SEED(42)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, model_name: str, epochs: int, best: str = ''):
    model_dir = os.path.join(MODELS_PATH, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = f"{best}{model_name}_epochs_{epochs}.pth"
    save_path = os.path.join(model_dir, model_file_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }, save_path)


def load_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, model_path: str):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def load_last_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, model_name: str):
    model_folder = os.path.join(MODELS_PATH, model_name)
    if os.path.exists(model_folder):
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
        if model_files:
            model_files.sort(key=lambda x: int(re.findall(r"epochs_(\d+)", x)[0])) # sort by epochs in filename
            last_model_file = model_files[-1] # load the model with the most epochs
            last_epochs = int(re.findall(r"epochs_(\d+)", last_model_file)[0])
            model_path = os.path.join(MODELS_PATH, model_name, last_model_file)
            load_model(model, optimizer, scheduler, model_path)
            print(f"Loaded model with {last_epochs} epochs.")
            return last_epochs
    return 0


def delete_last_best_model(model_name: str, prefix: str = 'BEST'):
    """Deletes the last saved best model file that contains 'BEST' in its name."""
    model_dir = os.path.join(MODELS_PATH, model_name)
    for file in os.listdir(model_dir):
        if prefix in file:
            os.remove(os.path.join(model_dir, file))


def format_metrics(metric_handler, train_metrics, val_metrics):
    log_metrics = {}
    formatted_metrics = []

    cumulative_val_loss = 0.0
    cumulative_val_acc = 0.0

    for name, train_value in train_metrics.items():
        val_value = val_metrics.get(name, "N/A")
        metric_type = "ACC" if metric_handler.metrics[name]['is_accuracy'] else "LOSS"

        formatted_metrics.append(f"{name} ({metric_type}): [Train={train_value:.4f}, Val={val_value:.4f}]")

        log_metrics[f"Train {name} ({metric_type})"] = train_value
        log_metrics[f"Val {name} ({metric_type})"] = val_value

        if metric_handler.metrics[name]['is_accuracy']:
            if val_value != "N/A":
                cumulative_val_acc += val_value
        else:
            if val_value != "N/A":
                cumulative_val_loss += val_value

    return formatted_metrics, log_metrics, cumulative_val_loss, cumulative_val_acc


def get_dataloaders(train_transform, val_transform, config) -> Tuple[DataLoader, DataLoader]:
    batch_size = config['batch_size']
    dataset_path = config['dataset_path']
    dataset_edition = config['dataset_edition']
    patches = config['input_shape']

    dataset_cls = h5VerSe if H5 in dataset_path else VerSe

    train_dataset = dataset_cls(root=dataset_path,
                                split='training',
                                transform=train_transform,
                                edition=dataset_edition,
                                download=True,
                                patches=patches)

    val_dataset = dataset_cls(root=dataset_path,
                              split='validation',
                              transform=val_transform,
                              edition=dataset_edition,
                              download=True,
                              patches=patches)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn)
    return train_loader, val_loader


def save_gflops(model_name, model, input_shape):
    model_dir = os.path.join(MODELS_PATH, model_name)
    os.makedirs(model_dir, exist_ok=True)
    gflops_file_name = "gflops.txt"
    save_path = os.path.join(model_dir, gflops_file_name)

    with open(save_path, "w") as file:
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            backend='pytorch',
            print_per_layer_stat=True,
            verbose=True,
            ost=file
        )
    return macs, params


def train(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, metric_handler: MetricHandler, target: str):
    model.train()
    metric_handler.reset()

    with tqdm(total=len(dataloader), desc='Training', unit='batch', leave=False) as pbar:
        for data in dataloader:
            pbar.set_postfix({'Processing': data['subject_id']})

            inputs = data[tio.IMAGE][tio.DATA]
            targets = data[target][tio.DATA]

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            total_loss = metric_handler.update(outputs, targets, accumulate_loss=True)

            if total_loss is not None:
                total_loss.backward()

            optimizer.step()

            pbar.update(1)
    return metric_handler.compute_metrics(len(dataloader))


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, metric_handler: MetricHandler, target: str):
    model.eval()
    metric_handler.reset()

    with tqdm(total=len(dataloader), desc='Validation', unit='batch', leave=False) as pbar:
        for data in dataloader:
            pbar.set_postfix({'Processing': data['subject_id']})

            inputs = data[tio.IMAGE][tio.DATA]
            targets = data[target][tio.DATA]

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)
            _ = metric_handler.update(outputs, targets, accumulate_loss=False)

            pbar.update(1)

    return metric_handler.compute_metrics(len(dataloader))



def train_loop(config: dict,
                     metric_handler: MetricHandler,
                     model: nn.Module, model_name: str,
                     optimizer: optim.Optimizer,
                     transforms_func: Callable,
                     visualize_func: Callable,
                     scheduler: optim.lr_scheduler.LRScheduler = None,
                     use_wandb: bool = False):

    input_shape = config['input_shape']
    macs, params = save_gflops(model_name, model, input_shape=(1, *input_shape))
    config['macs'] = '{:<30}  {:<8}'.format('Computational complexity: ', macs)
    config['params'] = '{:<30}  {:<8}'.format('Number of parameters: ', params)

    config['optimizer'] = optimizer.__class__.__name__
    config['architecture']: model.__class__.__name__

    if scheduler:
        config['scheduler'] = scheduler.__class__.__name__
        config['scheduler_params'] = scheduler.state_dict()

    wandb.init(project = 'master-thesis', name = model_name, group='studies', config = config) if use_wandb else None
    images_path = os.path.join(MODELS_PATH, model_name, 'images')
    os.makedirs(images_path, exist_ok=True)

    epochs = config['epochs'] + 1
    early_stopping = config['early_stopping']
    target = config['target']

    summary(model=model, input_size=(1, 1, *input_shape)) # B C H W D
    model.to(DEVICE)

    _, train_transform, val_transform, _, _  = transforms_func
    train_loader, val_loader = get_dataloaders(train_transform, val_transform, config)
    last_epoch = load_last_model(model, optimizer, scheduler, model_name)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    with tqdm(range(last_epoch+1, epochs), desc='Epochs', initial=last_epoch, total=epochs) as pbar:
        for epoch in pbar:

            train_metrics = train(model, train_loader, optimizer, metric_handler, target)
            val_metrics = validate(model, val_loader, metric_handler, target)
            formatted_metrics, log_metrics, val_loss, val_acc = format_metrics(metric_handler, train_metrics, val_metrics)

            if epoch % 1 == 0:
                visualize_func(subject=val_loader.dataset[0], output_path=os.path.join(images_path, f'{epoch}.png'),
                                    model=model, show=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                delete_last_best_model(model_name=model_name)
                save_model(model=model, optimizer=optimizer, scheduler=scheduler, model_name=model_name, epochs=epoch, best='BEST_')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping:
                print(f"Stopping early. Best validation loss: {best_val_loss}")
                break

            if scheduler:
                scheduler.step()

            pbar.set_postfix_str("; ".join(formatted_metrics))            
            wandb.log(log_metrics, step=epoch) if use_wandb else None
            delete_last_best_model(model_name=model_name, prefix='LAST_')
            save_model(model=model, optimizer=optimizer, scheduler=scheduler, model_name=model_name, epochs=epoch, best='LAST_')
            torch.cuda.empty_cache()

    wandb.finish() if use_wandb else None



def get_args():
    parser = argparse.ArgumentParser(description="Heatmap Training Configuration")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for saving/loading')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate for the optimizer')
    parser.add_argument('--input_shape', type=int, nargs=3, default=(64, 64, 128), help='Input shape for the model')
    parser.add_argument('--model', type=str, default='UNet', 
                        choices=['AttentionUNet3D', 'SegFormer3D', 'SwinUNetR', 'UNet'], 
                        help='Model architecture')
    parser.add_argument('--early_stopping', type=int, default=30, help='Early stopping criteria')
    parser.add_argument('--dataset_edition', type=int, default=19, help='VerSe dataset edition.')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use wandb for logging.')

    return parser.parse_args()


def get_model(args):
    if args.model == 'AttentionUNet3D':
        model = AttentionUNet3D(in_channels=1, out_channels=1).to(DEVICE)
    elif args.model == 'SegFormer3D':
        model = SegFormer3D(in_channels=1, num_classes=1).to(DEVICE)
    elif args.model == 'SwinUNetR':
        model = MonaiSwinUNetR(img_size=args.patch_size, in_channels=1, out_channels=1).to(DEVICE)
    elif args.model == 'UNet':
        model = MonaiUNet(in_channels=1, class_num=1).to(DEVICE)
    return model