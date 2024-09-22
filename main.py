import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from runners import train_test, train_test_private
from dataset import MedMNISTDataset, MedMNISTDataset_Rotated
from model import Net_28

from config import CONFIG

import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data():
    data = np.load(CONFIG['data_path'])

    if CONFIG['Split_Style'] == 'MedMNIST':
        train_images, train_labels = data['train_images'], data['train_labels']
        test_images, test_labels = data['test_images'], data['test_labels']
    else:
        all_images = np.concatenate((data['train_images'], data['test_images']), axis=0)
        all_labels = np.concatenate((data['train_labels'], data['test_labels']), axis=0)
        
        train_images, test_images, train_labels, test_labels = train_test_split(
            all_images, all_labels, test_size=CONFIG['split_ratio'], random_state=42
        )

    split = int(len(train_images) * CONFIG['split_ratio'])
    pretrain_images, finetune_images = train_images[:split], train_images[split:]
    pretrain_labels, finetune_labels = train_labels[:split], train_labels[split:]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    datasets = {
        'total': MedMNISTDataset(train_images, train_labels, transform=data_transform),
        'pretrain': MedMNISTDataset(pretrain_images, pretrain_labels, transform=data_transform),
        'pretrain_rotated': MedMNISTDataset_Rotated(pretrain_images, pretrain_labels, transform=data_transform),
        'finetune': MedMNISTDataset(finetune_images, finetune_labels, transform=data_transform),
        'test': MedMNISTDataset(test_images, test_labels, transform=data_transform)
    }

    return {name: DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
            for name, dataset in datasets.items()}

def save_model(model, path, is_private=False):
    state_dict = model.state_dict() if is_private else model.state_dict()
    torch.save(state_dict, path)

def load_model(model, path):
    state_dict = torch.load(path)
    model_dict = model.state_dict()
    
    new_state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items() if k.replace('_module.', '') in model_dict}
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    return model

def run_experiment(name, train_loader, test_loader, is_private=False):
    model = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    train_func = train_test_private if is_private else train_test
    acc, trained_model = train_func(model, train_loader, test_loader, task=CONFIG['task'], epochs=CONFIG['num_epochs'])
    save_model(trained_model, f"weights/{name}.pth", is_private)
    return acc

def run_experiment_multiple_times(experiment_func, times=CONFIG['Top-K']):
    results = []
    for _ in range(times):
        result = experiment_func()
        results.append(result)
    return results

def calculate_stats(results):
    avg_results = {}
    std_results = {}
    for key in results[0].keys():
        values = [result[key] for result in results]
        avg_results[key] = np.mean(values)
        std_results[key] = np.std(values)
    return avg_results, std_results

def run_base_experiments():
    def single_run():
        loaders = load_and_prepare_data()
        results = {}

        experiments = [
            ("total", "total", False),
            ("pretrain", "pretrain", False),
            ("finetune", "finetune", False),
            ("pretrain_rotated", "pretrain_rotated", False),
            ("private_total", "total", True),
            ("private_pretrain", "pretrain", True),
            ("private_finetune", "finetune", True),
            ("private_pretrain_rotated", "pretrain_rotated", True)
        ]

        for name, loader_name, is_private in experiments:
            print(f"Loader: {loader_name}, Private: {is_private}")
            results[name] = run_experiment(name, loaders[loader_name], loaders['test'], is_private)

        return results

    return run_experiment_multiple_times(single_run)

def run_pretrain_finetune_experiments():
    def single_run():
        loaders = load_and_prepare_data()
        results = {}

        experiments = [
            ("pretrain_normal__finetune_normal", "pretrain.pth", False),
            ("pretrain_normal__finetune_private", "pretrain.pth", True),
            ("pretrain_private__finetune_normal", "private_pretrain.pth", False),
            ("pretrain_private__finetune_private", "private_pretrain.pth", True),
            ("pretrain_normalrotated__finetune_normal", "pretrain_rotated.pth", False),
            ("pretrain_normalrotated__finetune_private", "pretrain_rotated.pth", True),
            ("pretrain_privaterotated__finetune_normal", "private_pretrain_rotated.pth", False),
            ("pretrain_privaterotated__finetune_private", "private_pretrain_rotated.pth", True)
        ]

        for name, pretrain_weights, is_private in experiments:
            print(f"Pretrain: {pretrain_weights}, Finetune: {name[17:]}, Private: {is_private}")
            model = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
            model = load_model(model, f"weights/{pretrain_weights}")
            
            train_func = train_test_private if is_private else train_test
            acc, trained_model = train_func(model, loaders['finetune'], loaders['test'], task=CONFIG['task'], epochs=CONFIG['num_epochs'])
            
            save_model(trained_model, f"weights/{name}.pth", is_private)
            results[name] = acc
        
        return results

    return run_experiment_multiple_times(single_run)

def print_results_table(results, title):
    avg_results, std_results = calculate_stats(results)
    
    # Create a DataFrame
    df = pd.DataFrame({
        "Experiment": avg_results.keys(),
        "Accuracy (Mean)": [f"{value:.4f}" for value in avg_results.values()],
        "Accuracy (Std)": [f"{value:.4f}" for value in std_results.values()]
    })
    
    # Print the table
    print(f"\n{title}")
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(f"results/{title}.csv", index=False)

if __name__ == '__main__':
    base_results = run_base_experiments()
    pretrain_finetune_results = run_pretrain_finetune_experiments()

    print_results_table(base_results, f"Base_{CONFIG['data']}")
    print_results_table(pretrain_finetune_results, f"Compound_{CONFIG['data']}")