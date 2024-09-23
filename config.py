import torch

data_path =  '/home/localssk23/.medmnist/'

def generate_data_field(datasets):
    return "_".join(dataset.split("/")[-1].split(".")[0] for dataset in datasets)

CONFIG = {
    "batch_size": 32,
    "num_channels": 3,
    "num_epochs": 2,
    "task": "multi-class",

    "datasets": [f"{data_path}pneumoniamnist.npz", f"{data_path}breastmnist.npz", f"{data_path}retinamnist.npz"],

    "split_ratio": 0.1,
    "Top-K": 2,
    "Split_Style": 'MedMNIST',
    #? 'MedMNIST' - Use Existing MNIST Splits
    #? 'Custom' - Use Custom Splits

    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Automatically generate the "data" field
CONFIG["data"] = generate_data_field(CONFIG["datasets"])