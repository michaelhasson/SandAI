import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision
from torchvision.transforms import Lambda, Normalize, RandomVerticalFlip, RandomHorizontalFlip, ToTensor, Resize, ColorJitter, RandomAutocontrast, RandomInvert, RandomRotation, Compose
from torchvision.transforms.v2 import RandomEqualize
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import time
from PIL import Image
import os
import random
from collections import Counter

# Image transformation
img_height = 224
img_width = 224

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomEqualize(p=1),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, filenames, transform=None):
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path)

        # Ensure the image is converted to RGB format if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

def make_predictions(sample, imagedataset, imagedl, model_path, csv_path, confidence_threshold=0.75):
    print(f'Predicting on sample {sample}')
    
    # Device configuration
    device = torch.device("cpu")
    
    # Set class names
    class_names = {0: 'Eolian', 1: 'Glacial', 2: 'Beach', 3: 'Fluvial'}
    
    # Load dataset/loader
    dataset = imagedataset 
    filenames = dataset.filenames
    loader = imagedl

    # Load model
    model = models.resnet50()
    # Set number of classes
    num_classes = 4
    model.fc = torch.nn.Linear(2048, num_classes)

    # Load checkpoint
    cp_filepath = model_path
    checkpoint = torch.load(cp_filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    # Switch to eval mode
    model.eval()

    # Set the CSV export path 
    csv_path = csv_path

    confidence_threshold = confidence_threshold

    predictions, probabilities, filtered_filenames_list = [], [], []

    with torch.no_grad():
        for idx, (images) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            max_probs, _ = torch.max(probs, dim=1)
            threshold_tensor = torch.full_like(max_probs, confidence_threshold)

            mask = max_probs > threshold_tensor

            filtered_preds = preds[mask]

            filtered_filenames = [imagedataset.filenames[idx] for i, keep in enumerate(mask) if keep]  # Get the filename for this specific batch index
            filtered_filenames_list.extend(filtered_filenames)

            filtered_probabilities = probs[mask].tolist()

            predictions.extend([pred.item() for pred in filtered_preds])
            probabilities.extend(filtered_probabilities)

    # Convert class numbers to class names
    predictions_named = [class_names[pred] for pred in predictions]

    # Count number of predictions in each class and percentages of total
    count_dict = Counter(predictions_named)
    total_predictions = len(predictions_named)
    class_counts = {f'{name}_count': count_dict[name] for name in class_names.values()}
    class_percentages = {f'{name}_percentage': (count_dict[name] / total_predictions * 100) if total_predictions > 0 else 0 for name in class_names.values()}

    # Create df for exporting 
    main_data = {
        'filename': filtered_filenames_list,
        'predictions': predictions_named,
        'probabilities': probabilities
    }
    main_df = pd.DataFrame(main_data)

    # Create summary data
    summary_data = {
        'Eolian_count': [class_counts.get('Eolian_count', '')],
        'Glacial_count': [class_counts.get('Glacial_count', '')],
        'Beach_count': [class_counts.get('Beach_count', '')],
        'Fluvial_count': [class_counts.get('Fluvial_count', '')],
        'Eolian_percentage': [class_percentages.get('Eolian_percentage', '')],
        'Glacial_percentage': [class_percentages.get('Glacial_percentage', '')],
        'Beach_percentage': [class_percentages.get('Beach_percentage', '')],
        'Fluvial_percentage': [class_percentages.get('Fluvial_percentage', '')]
    }

    
    # Expand the summary data to have the same number of rows as main_df
    summary_df = pd.DataFrame(summary_data)
    for col in summary_df.columns:
        summary_df[col] = pd.concat([summary_df[col], pd.Series([''] * (len(main_df) - 1))], ignore_index=True)
        
    # Concatenate the dataframes horizontally
    final_df = pd.concat([main_df, summary_df], axis=1)

    final_df.to_csv(csv_path, index=False)
    print(f'Saved CSV at: {csv_path}')

def display_random_images(imagedataset, num_images_to_display=9):
    length_of_dataset = len(imagedataset)
    
    if num_images_to_display > length_of_dataset:
        raise ValueError("The count of random images must not exceed the length of the dataset.")
    
    random_indices = random.sample(range(length_of_dataset), num_images_to_display)
    
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    
    for i, idx in enumerate(random_indices):
        image = imagedataset[idx]
        image_np = transforms.ToPILImage()(image)
        image_np = image_np.convert('RGB')
        image_np = np.array(image_np)
        
        axs[i // 3, i % 3].imshow(image_np)
        axs[i // 3, i % 3].axis('off')
    
    plt.show()
