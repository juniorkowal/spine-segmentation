# Spine Segmentation

This repository is a part of my **Master's Thesis**, focusing on **binary vertebrae segmentation** from CT images. The project is in an **experimental stage** and might contain bugs or bloated code since many features and techniques were tested during its development. Despite this, most unnecessary parts have been removed from this repo.

## Disclaimer
The project is experimental and not fully optimized. It may contain bugs, and the implementation is primarily designed for constrained hardware (like a GTX 1050 Ti).

## Setup
Developed using Python 3.10/3.11, CUDA 12.1, and PyTorch 2.2.1.

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## Project Overview

The project utilizes the **VerSe19** or **VerSe20** dataset for vertebrae segmentation. My experiments were conducted 
on **VerSe19** under constrained hardware conditions. The project uses **torchio** for data preprocessing, with custom 
preprocessing classes specifically tailored for the VerSe dataset. 

### Workflow:
1. **Preprocessing**: Preprocess, convert and compress the VerSe dataset into the H5 format using `h5py` and `hdf5plugin` for efficient memory usage (e.g. do resampling to 1mm for CT images beforehand).
2. **Training**:
   - **Heatmap (Spine Localization)**: Train a heatmap model for spine localization with the following command:
     ```
     python training/train_heatmap.py --data_path <dataset_path> --model_name <your_model_name>
     ```
   - **Segmentation**: Train a segmentation model using:
     ```
     python training/train_segmentation.py --data_path <dataset_path> --model_name <your_model_name>
     ```

Both training scripts share the same arguments. Here is a basic outline of the arguments:

### Training Arguments
```python
def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")

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
```

### Heatmap vs Segmentation
**Heatmap**: For spine localization, I resized CT images to 64x64x128 as a preprocessing step.

**Segmentation**: For segmentation, I use random patches (e.g. 128x128x128) extracted directly from the disk (using the H5 format). This avoids loading the whole image into memory.

### Logging & Visualization

- You can optionally use **Weights & Biases (wandb)** to log metrics during training.
- Training saves both the **last model** and the **best model**.
- Slices of images are saved after each epoch for visual tracking of model performance.
- After training, **test set metrics** are evaluated to provide the final performance results.

### Multiclass Segmentation
The code can be easily adapted for multiclass segmentation tasks by changing models input shape and mask remapping in preprocessing step.

### Important Notes
The current implementation is tested **only with batch size = 1**. It may not work with larger batch sizes without additional modifications.

### Project Structure
```bash
├── data_transforms
│   ├── heatmap_transforms.py       # Data transforms for heatmap generation
│   ├── segmentation_transforms.py  # Data transforms for segmentation tasks
│   └── classes/                    # Custom transformation classes compatible with torchio
├── metrics
│   ├── losses.py                   # Loss functions
│   ├── metric_handler.py           # Metric handling class
├── networks
│   ├── UNet.py                     # Various model architectures (UNet, AttentionUNet, etc.)
├── training
│   ├── scripts.py                  # Basic training utilities
│   ├── train_heatmap.py            # Training script for heatmap models
│   ├── train_segmentation.py       # Training script for segmentation models
├── utils
│   ├── constants.py                # Global constants (like DEVICE)
│   ├── misc.py                     # Miscellaneous utilities
│   ├── statistics.py               # Needs refactoring; analysis utilities
│   ├── visualize.py                # Slice plotting from model predictions
├── verse_utils                     # Scripts from VerSe dataset GitHub
├── dataloader.py                   # Dataloaders for VerSe and h5VerSe datasets
├── preprocessing.py                # Preprocess raw data into H5 format
```
### Dataset
The project uses the **VerSe19** and **VerSe20** datasets.

- The dataset will be downloaded automatically if it is not found in the specified path.
- Use the `VerSe` class in `dataloader.py` to load the dataset.

### Preprocessing
Run the preprocessing script (`preprocessing.py`) to convert raw CT data into the H5 format for efficient memory handling.

### Model Architectures
This repository includes various segmentation model architectures, such as:

- **UNet**
- **AttentionUNet3D**
- **SegFormer3D**
- **SwinUNetR**

You can specify the desired architecture in the training arguments.

**TODO:**
- Refactor the **statistics** script
- Implement an **inference** script
- Add more **model** architectures
- Enhance **visualization** capabilities to include 3D representations instead of just slices
