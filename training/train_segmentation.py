import os
import torch.nn as nn
import wandb
from data_transforms import binary_segmentation_transforms
from metrics.losses import *
from metrics.metric_handler import MetricHandler
from networks import *
from prodigyopt import Prodigy
from torchinfo import summary
from tqdm.autonotebook import tqdm
# from tqdm.notebook import tqdm
from training.scripts import *
from utils.constants import DEVICE, MODELS_PATH
from utils.visualize import visualize_spine_segmentation
from data_transforms.classes import *
from preprocessing import basic_dataset_prepro


def get_segmentation_prepro():
    # payer c. kinda
    prep1 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.Clamp(out_min = -1024),
        tio.Blur(std = 0.75),
        MaskCutout(threshold=0.5),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # with noise
    prep2 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        MaskCutout(threshold=0.5),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # denoised by sn2n
    prep3 = None
    # prep3 = tio.Compose([
    #     tio.ToCanonical(),
    #     tio.Resample(1), # 8
    #     MaskCutout(threshold=0.5),
    #     PadDimTo(size=(96,96,128)),
    #     SN2NDenoise(model=SN2NUNet(),
    #                 optimizer=torch.optim.Adam(SN2NUNet().parameters(), lr = 2e-4, betas = (0.5, 0.999)),
    #                 model_path='/home/manis/Desktop/magisterka/vertebrae-segmentation/trained_models/SN2N_DENOISING/BEST_SN2N_DENOISING_epochs_130.pth',
    #                 patch_size=(96,96,128),
    #                 patch_overlap=(48,48,64),
    #                 model_act=F.sigmoid,
    #                 device='cuda',
    #                 batch_size=1,
    #                 ),
    #     tio.RescaleIntensity(out_min_max=(0,1)),

    # ])

    # nlm3denoise test
    prep4 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1), # 8
        tio.RescaleIntensity(out_min_max=(0,1)),
        MaskCutout(threshold=0.5),
        AlgorithmicDenoise(mode='nlm3d'),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    # bm4d denoise
    prep5 = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.RescaleIntensity(out_min_max=(0,1)),
        MaskCutout(threshold=0.5),
        AlgorithmicDenoise(mode='bm4d'),
        tio.RescaleIntensity(out_min_max=(0,1)),
    ])

    return prep1, prep2, prep3, prep4, prep5



def train_segmentation(data_path, model, model_name, transform_func, loss_class, patch_size=(64,64,64)):
    # DEVICE = 'cpu'
    spine_seg_metrics = MetricHandler()
    spine_seg_metrics.add_metric(name=loss_class.__class__.__name__, metric_fn=loss_class, is_accuracy=False)

    eval_metrics = {'JaccardIndex': Jaccard(),
                    'F1Score': F1Score(),
                    'Recall': Recall(),
                    'Precision': Precision(),
                    # 'Hausdorff': Hausdorff()
    }

    for name, fn in eval_metrics.items():
        spine_seg_metrics.add_metric(name=name, metric_fn=fn.to(DEVICE), is_accuracy=True)

    class_num = 1
    shape = patch_size
    # shape = (128,128,128)
    # shape = (64,64,64)
    overlap = (shape[0]//2,shape[1]//2,shape[2]//2)

    epochs = 200
    # epochs = 1

    lr = 1.

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
        'target': tio.LABEL,
    }

    train_loop(config=config,
                     metric_handler=spine_seg_metrics,
                     model=model,
                     model_name=model_name,
                     transforms_func=transform_func,
                     visualize_func=visualize_spine_segmentation,
                     optimizer=optimizer,
                     scheduler=scheduler)

    # test thing
    _, _, test_transform, _, _ = transform_func(shape)
    test_dataset = h5VerSe(root=data_path, split='test', transform=test_transform, edition=19, download=True)

    model_folder = os.path.join(MODELS_PATH, model_name)
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
    best_model = None
    for m in model_files:
        if 'BEST' in m:
            best_model = m

    print('\nRunning model through test set.',best_model,'\n')
    load_model(model, optimizer, scheduler, model_path=os.path.join(MODELS_PATH, model_name, best_model))
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model.to(DEVICE)
    # CHECK MODEL METRICS ON TEST SET

    spine_seg_metrics.add_metric(name='Hausdorff', metric_fn=Hausdorff().to(DEVICE), is_accuracy=True)

    with torch.no_grad():

        model.eval()
        spine_seg_metrics.reset()

        with tqdm(total=len(loader), desc='Test', unit='batch', leave=False) as pbar:
            for i, data in enumerate(loader):
                # if i > 0:
                #     continue
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
                # print(torch.unique(targets.squeeze(0)), torch.unique(outputs))
                # print(torch.unique(targets.squeeze(0)-outputs))
                # diff = targets.squeeze(0)-outputs
                # diff[diff == -1.0] = 1.0

                # tio.Subject(
                #     # image = tio.ScalarImage(tensor = inputs.squeeze(0)),
                #             # target = tio.LabelMap(tensor=targets.squeeze(0)),
                #             # prediction = tio.LabelMap(tensor=outputs)).plot()
                #             prediction = tio.ScalarImage(tensor=diff)).plot()

                # tio.Subject(
                #     # image = tio.ScalarImage(tensor = inputs.squeeze(0)),
                #             target = tio.LabelMap(tensor=targets.squeeze(0)),
                #             # prediction = tio.LabelMap(tensor=outputs)
                # ).plot()
                pbar.update(1)

        best_result = spine_seg_metrics.compute_metrics(len(loader))

    result_path = os.path.join('trained_models', model_name, 'test_set_result.txt')
    with open(result_path, mode='w') as f:
        f.writelines(str(best_result))


if __name__ == "__main__":

    losses_testing = [
        BinaryCrossEntropyWithLogits(),
                       DiceLoss(),
                      DiceCELoss()
                      ]
    preprocessing_testing = get_segmentation_prepro()
    prepro_names = ['seg_payerc', 'seg_noisy', 'seg_sn2n', 'seg_nlm3', 'seg_bm4d']

    # for i, prepro in enumerate(preprocessing_testing):
    #     # if i==2 or i==4:
    #     # if i!=2:
    #     #     continue
    #     basic_dataset_prepro(out_name=prepro_names[i],transform=prepro)

    # # CHECK BEST DENOISING PREPROCESSING
    # for i, prepro in enumerate(preprocessing_testing):
    #     # if prepro_names[i] != 'seg_sn2n':
    #     #     continue
    #     name = prepro_names[i]
    #     train_segmentation(data_path=f'datasets/{name}/{name}.h5', model=MonaiUNet(), model_name=name, transform_func=binary_segmentation_transforms, loss_class=losses_testing[2])

    # bm4d has best boundary hausdorf loss and segmentation loss but lower precision, still bm4d for the win
    # CHECK LOSSES
    # for loss in losses_testing:
    #     model_name = 'seg_' + loss.__class__.__name__
    #     name = prepro_names[-1]
    #     train_segmentation(data_path=f'datasets/{name}/{name}.h5', model=MonaiUNet(), model_name=model_name, transform_func=binary_segmentation_transforms, loss_class=loss)
        # train_segmentation(data_path=f'/kaggle/input/seg-bm4d/{name}.h5', model=MonaiUNet(), model_name=model_name, transform_func=binary_segmentation_transforms, loss_class=loss)

    # CHECK MODELS
    from networks._unused.mixswinunetr.mixswinunetr import SwinMUNETR
    from networks._unused.lambdaunet import LambdaUNet
    # from networks._unused.nnformer.nnFormer_tumor import nnFormer
    loss = DiceLoss()
    models = [
        # (AttentionUNet3D(1,1), (128,128,128)), # possible (128,128,128)
        # (MUNet(), (64,64,64)), #b (64,64,64)
        # (SegFormer3D(in_channels=1, num_classes=1), (128,128,128)), # possible (128,128,128)
        # (SwinMUNETR(img_size=(64,64,64), in_channels=1, out_channels=1), (64,64,64)), #b
        # (MonaiSwinUNetR(input_size=(64,64,64)), (64,64,64)), #b (96,96,96)
        # (MonaiVNet(), (96,96,96)), #b (96,96,96)
        # (MonaiUNet(), (128,128,128)),
        # (UNet3Plus_modified(), (32,32,32)), #b (32,32,32)
        ######(UNet3Plus_modified(), (96,96,96)), #b (32,32,32)

        # MonaiUNetR(input_size=(96,96,96)), #b
        (LambdaUNet(), (128,128,128))
        ]
    shapes = []
    for model, patch_size in models:
        # summary(model, input_size=(1,1,128,128,128))
        name = model.__class__.__name__.replace('Monai', '')
        name += '0LAMBDA_LAYERNORM_'
        print(name)
        train_segmentation(data_path='datasets/seg_bm4d/seg_bm4d.h5', model=model, model_name=name, transform_func=binary_segmentation_transforms, loss_class=loss, patch_size=patch_size)
        # train_segmentation(data_path='/kaggle/input/seg-bm4d/seg_bm4d.h5', model=model, model_name=name, transform_func=binary_segmentation_transforms, loss_class=loss, patch_size=patch_size)
        # exit()
        torch.cuda.empty_cache()


    # model = MonaiUNet(in_channels=1, class_num = 1).to(DEVICE)
    # model = SegFormer3D(num_classes=1, in_channels=1).to(DEVICE)
    # model = MonaiUNet()

    # data_path = 'datasets/verse19-bboxcutout-basic-prepro-resample1/verse19-bboxcutout-basic-prepro-resample1.h5'
    # model_name = 'testSEGFORMER'

    # train_segmentation(data_path, model, model_name, binary_segmentation_transforms)






    # # MODEL 2 SPINE SEGMENTATION BY RANDOM PATCHES
    # spine_seg_metrics = MetricHandler()
    # spine_seg_metrics.add_metric(name='DiceLoss', metric_fn=DiceLoss(), is_accuracy=False)

    # eval_metrics = {'JaccardIndex': Jaccard(),
    #                 'F1Score': F1Score(),
    #                 'Recall': Recall(),
    #                 'Precision': Precision(),
    #                 'Hausdorff': Hausdorff()
    # }

    # for name, fn in eval_metrics.items():
    #     spine_seg_metrics.add_metric(name=name, metric_fn=fn.to(DEVICE), is_accuracy=True)

    # data_path = 'datasets/verse19-bboxcutout-basic-prepro-resample1/verse19-bboxcutout-basic-prepro-resample1.h5'

    # # shape = (128,128,128)
    # shape = (64,64,64)
    # # shape = (96,96,96)
    # # shape = (96,96,128)

    # model_name = 'testSEGFORMER'

    # class_num = 1
    # in_channels = 1

    # model = SegFormer3D(num_classes=class_num, in_channels=in_channels)

    # epochs = 200
    # lr = 1.
    # optimizer = Prodigy(params=model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # config = {
    #     'learning_rate': lr,
    #     'epochs': epochs,
    #     'batch_size': 1,
    #     'input_shape': shape,
    #     'dataset_path': data_path,
    #     'dataset_edition': '19',
    #     'early_stopping':30,
    #     'target': tio.LABEL,
    # }

    # train_loop(config=config,
    #                  metric_handler=spine_seg_metrics,
    #                  model=model,
    #                  model_name=model_name,
    #                  transforms_func=binary_segmentation_transforms,
    #                  visualize_func=visualize_spine_segmentation,
    #                  optimizer=optimizer,
    #                  scheduler=scheduler)
