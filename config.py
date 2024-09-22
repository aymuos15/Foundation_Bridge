import torch

CONFIG = {
    "batch_size": 32,
    "task": "multi-class",
    "num_classes": 5,
    "num_channels": 3,
    "num_epochs": 2,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "data_path": '/home/localssk23/.medmnist/retinamnist.npz',
    "data": "retinamnist",
    "split_ratio": 0.1,
    "Top-K": 2,
    "Split_Style": ''
}
