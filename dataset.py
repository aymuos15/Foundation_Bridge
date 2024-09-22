import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

import random

class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class MedMNISTDataset_Rotated(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.rotation_angles = [90, 180]  # Define possible rotation angles

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Ensure the image has a channel dimension
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        # Convert to PIL Image for rotation
        pil_image = Image.fromarray(image.squeeze().astype(np.uint8))

        # Randomly select a rotation angle
        rotation_angle = random.choice(self.rotation_angles)
        
        # Rotate the image
        rotated_image = pil_image.rotate(rotation_angle)

        if self.transform:
            rotated_image = self.transform(rotated_image)
        else:
            # If no transform is provided, convert to tensor manually
            rotated_image = torch.from_numpy(np.array(rotated_image)).float()
            if rotated_image.ndim == 2:
                rotated_image = rotated_image.unsqueeze(0)
        
        # Create a new label based on the rotation angle
        rotation_label = torch.tensor(self.rotation_angles.index(rotation_angle))

        return rotated_image, rotation_label
    
# # To use Multiple Dataset Potentially
# def combine_datasets(*datasets):
#     """
#     Combine multiple datasets, adjusting labels to ensure uniqueness across all datasets.
    
#     Args:
#     *datasets: Variable number of tuples, each containing (images, labels) for a dataset.
    
#     Returns:
#     combined_images: Numpy array of combined images from all datasets.
#     combined_labels: Numpy array of combined and adjusted labels from all datasets.
#     """
#     combined_images = []
#     combined_labels = []
#     label_offset = 0

#     for i, (images, labels) in enumerate(datasets):
#         combined_images.append(images)
        
#         # Adjust labels for uniqueness
#         adjusted_labels = labels + label_offset
#         combined_labels.append(adjusted_labels)
        
#         # Update label offset for the next dataset
#         label_offset += len(np.unique(labels))
        
#         # Print information about this dataset
#         print(f'Dataset {i+1} labels:', np.unique(adjusted_labels))
#         print(f'Dataset {i+1} shape:', images.shape)

#     # Combine all images and labels
#     combined_images = np.concatenate(combined_images, axis=0)
#     combined_labels = np.concatenate(combined_labels, axis=0)

#     print('\nCombined dataset:')
#     print('Labels:', np.unique(combined_labels))
#     print('Shape:', combined_images.shape)

#     return combined_images, combined_labels

# a = np.load('/home/soumya/.medmnist/retinamnist.npz')
# c = np.load('/home/soumya/.medmnist/bloodmnist.npz')

# combined_images, combined_labels = combine_datasets((a['train_images'], a['train_labels']), (c['train_images'], c['train_labels']))
# test_combined_images, test_combined_labels = combine_datasets((a['test_images'], a['test_labels']), (c['test_images'], c['test_labels']))