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
    
# To use Multiple Dataset Potentially
def combine_datasets(data_paths):
    """
    Combine multiple datasets from a list of file paths, adjusting labels to ensure uniqueness across all datasets.
    Handles both training and test data.
    
    Args:
    data_paths: List of strings, each containing the file path to a .npz dataset.
    
    Returns:
    combined_train_images: Numpy array of combined training images from all datasets.
    combined_train_labels: Numpy array of combined and adjusted training labels from all datasets.
    combined_test_images: Numpy array of combined test images from all datasets.
    combined_test_labels: Numpy array of combined and adjusted test labels from all datasets.
    """

    #! Do not mix one channel and multi channel datasets. Example breastmnist and retinamnist

    combined_train_images = []
    combined_train_labels = []
    combined_test_images = []
    combined_test_labels = []
    label_offset = 0

    for path in data_paths:
        # Load the dataset
        data = np.load(path)
        
        # Extract train images and labels
        train_images = data['train_images']
        train_labels = data['train_labels']
        
        # Extract test images and labels
        test_images = data['test_images']
        test_labels = data['test_labels']

        combined_train_images.append(train_images)
        combined_test_images.append(test_images)

        # Adjust labels for uniqueness
        adjusted_train_labels = train_labels + label_offset
        adjusted_test_labels = test_labels + label_offset
        
        combined_train_labels.append(adjusted_train_labels)
        combined_test_labels.append(adjusted_test_labels)

        # Update label offset for the next dataset
        label_offset += len(np.unique(train_labels))

    # Combine all images and labels
    combined_train_images = np.concatenate(combined_train_images, axis=0)
    combined_train_labels = np.concatenate(combined_train_labels, axis=0)
    combined_test_images = np.concatenate(combined_test_images, axis=0)
    combined_test_labels = np.concatenate(combined_test_labels, axis=0)

    return combined_train_images, combined_train_labels, combined_test_images, combined_test_labels