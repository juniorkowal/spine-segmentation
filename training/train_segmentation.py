import pytest
pytest.main([__file__])

import os

from data_transforms import binary_segmentation_transforms
from data_transforms.classes import *
from metrics.losses import *
from metrics.metric_handler import MetricHandler
from networks import *
from preprocessing import dataset_preprocessing
from prodigyopt import Prodigy
from tqdm.autonotebook import tqdm
from training.scripts import *
from utils.constants import DEVICE, MODELS_PATH
from utils.visualize import visualize_spine_segmentation



def train_segmentation(args):
    model = get_model(args)

    spine_seg_metrics = MetricHandler()
    spine_seg_metrics.add_metric(name='DiceLoss', metric_fn=DiceLoss(), is_accuracy=False)

    eval_metrics = {'JaccardIndex': Jaccard(),
                    'F1Score': F1Score(),
                    'Recall': Recall(),
                    'Precision': Precision(),
                    # 'Hausdorff': Hausdorff()
    }

    for name, fn in eval_metrics.items():
        spine_seg_metrics.add_metric(name=name, metric_fn=fn.to(DEVICE), is_accuracy=True)

    shape = args.input_shape
    overlap = (shape[0]//2, shape[1]//2, shape[2]//2)

    optimizer = Prodigy(params=model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'input_shape': args.input_shape,
        'dataset_path': args.data_path,
        'dataset_edition': args.dataset_edition,
        'early_stopping': args.early_stopping,
        'target': tio.LABEL,
        'model_name': args.model_name,
        'run_id': args.run_id
    }

    transform_func = binary_segmentation_transforms(args.input_shape)
    train_loop(config=config,
                     metric_handler=spine_seg_metrics,
                     model=model,
                     model_name=args.model_name,
                     transforms_func=transform_func,
                     visualize_func=visualize_spine_segmentation,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     use_wandb=args.use_wandb)

    # test
    test_transform = transform_func[2]
    test_dataset = h5VerSe(root=args.data_path, split='test', transform=test_transform, edition=args.dataset_edition, download=True)

    best_model = next((f for f in os.listdir(os.path.join(MODELS_PATH, config['model_name'])) if 'BEST' in f), None)

    print('\nRunning model through test set.',best_model,'\n')
    load_model(model, optimizer, scheduler, model_path=os.path.join(MODELS_PATH, args.model_name, best_model))
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model.to(DEVICE)
    spine_seg_metrics.add_metric(name='Hausdorff', metric_fn=Hausdorff().to(DEVICE), is_accuracy=True)

    with torch.no_grad():
        model.eval()
        spine_seg_metrics.reset()

        with tqdm(total=len(loader), desc='Test', unit='batch', leave=False) as pbar:
            for i, data in enumerate(loader):
                pbar.set_postfix({'Processing': data['subject_id']})

                inputs = data[tio.IMAGE][tio.DATA]
                targets = data[tio.LABEL][tio.DATA]

                grid_sampler = tio.inference.GridSampler(
                    tio.Subject(image=tio.ScalarImage(tensor=inputs.squeeze(0))),
                    shape,
                    overlap,
                )
                patch_loader = DataLoader(grid_sampler, batch_size=1)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')

                with tqdm(total=len(patch_loader), desc='Patch assembling', unit='patch', leave=False) as patch_pbar:
                    for patches_batch in patch_loader:
                        input_tensor = patches_batch[tio.IMAGE][tio.DATA].to(DEVICE)
                        locations = patches_batch[tio.LOCATION]
                        outputs = model(input_tensor)
                        # outputs = F.sigmoid(outputs)
                        aggregator.add_batch(outputs, locations)
                        patch_pbar.update(1)

                outputs = F.sigmoid(aggregator.get_output_tensor())#.cpu()
                outputs = (outputs > 0.5).float()

                _ = spine_seg_metrics.update(outputs.unsqueeze(0).to(DEVICE), targets.to(DEVICE), accumulate_loss=False)
                pbar.update(1)

        best_result = spine_seg_metrics.compute_metrics(len(loader))

    result_path = os.path.join('trained_models', args.model_name, 'test_set_result.txt')
    with open(result_path, mode='w') as f:
        f.writelines(str(best_result))



if __name__ == "__main__":
    SET_SEED(42)
    args = get_args()
    train_segmentation(args)
