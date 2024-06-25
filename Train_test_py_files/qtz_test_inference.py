import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
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

device = torch.device("cpu")
if torch.cuda.is_available():    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

train_path = r"D:\Michael\Quartz_classifier\data\4_4\train"
val_path = r"D:\Michael\Quartz_classifier\data\4_4\val"
test_path = r"D:\Michael\Quartz_classifier\data\4_4\test"


img_height = 224
img_width = 224

num_classes = 4

transform = Compose([
    Resize((img_height, img_width)),
    RandomEqualize(p=1),
    ToTensor(),
])

def try_load(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
        return None
    
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)    
test_dataset = datasets.ImageFolder(root=test_path, transform=transform, loader=try_load)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

class_sample_count = np.unique(train_dataset.targets, return_counts=True)[1]
criterion = nn.CrossEntropyLoss()

def predictions_stats(split, best_stat):
    
    print(f'Predicting with best {best_stat} on {split} set')
    
    if split == 'train':
        dataset = train_dataset
        loader = train_loader
    elif split == 'val':
        dataset = val_dataset
        loader = val_loader
    elif split == 'test':
        dataset = test_dataset
        loader = test_loader
    else:
        return

    # Load model
    cp_filepath = r"D:\Michael\Quartz_classifier\Final paper materials\model\qtz_final_model.pth"


    checkpoint = torch.load(cp_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Switch to eval mode
    model.eval()

    predictions, actuals, filenames, probabilities = [], [], [], []
    low_confidence_predictions, low_confidence_actuals, low_confidence_filenames, low_confidence_probabilities = [], [], [], []

    confidence_threshold = 0.00

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            if idx % 5000 == 0:
                print(f'predicting on {split} set, example ', idx)
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Keep predictions with confidence greater than the threshold
            max_probs, _ = torch.max(probs, dim=1) 
            threshold_tensor = torch.full_like(max_probs, confidence_threshold)
            mask = max_probs > threshold_tensor

            filtered_preds = preds[mask]

            filtered_labels = labels[mask]
            filtered_filenames = [dataset.samples[idx][0] for i, keep in enumerate(mask) if keep]
            filtered_probabilities = probs[mask].tolist()
            
            predictions.extend([pred.item() for pred in filtered_preds])
            actuals.extend([label.item() for label in filtered_labels])
            filenames.extend(filtered_filenames)
            probabilities.extend(filtered_probabilities)


            # Do the same for low-confidence images to understand where the model fails
            # Identify low confidence predictions
            low_confidence_mask = ~mask
            low_confidence_labels = labels[low_confidence_mask]
            low_confidence_filenames.extend([dataset.samples[idx][0] for i, keep in enumerate(low_confidence_mask) if keep])
            low_confidence_probabilities = probs[low_confidence_mask].tolist()
            low_confidence_preds = preds[low_confidence_mask]

            low_confidence_predictions.extend([pred.item() for pred in low_confidence_preds])
            low_confidence_actuals.extend([label.item() for label in low_confidence_labels])
            # low_confidence_filenames.extend(low_confidence_filenames)
            low_confidence_probabilities.extend(low_confidence_probabilities)
               
        # Save high confidence predictions
        df = pd.DataFrame({'filename': filenames, 'predictions': predictions, 'actuals': actuals, 'probability': probabilities})
        df.to_csv(f'D:/Michael/Quartz_classifier/Final paper materials/final_predictions/Train_val_test_preds/{split}_predictions_with_confidence_threshold_final.csv', index=False)

        print(f'Saved csv at: D:/Michael/Quartz_classifier/Final paper materials/final_predictions/Train_val_test_preds/{split}_predictions_with_confidence_threshold_final.csv')

    all_predictions = np.array(predictions)
    all_actuals = np.array(actuals)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(all_actuals, all_predictions, labels=range(num_classes))

    # Print or visualize the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate class-wise metrics
    classwise_metrics = {}
    for class_idx in range(num_classes):
        true_positive = conf_matrix[class_idx, class_idx]
    
    print('class_idx', class_idx)
    
    per_class_accuracy = []
    class_labels = [0, 1, 2, 3]

    for label in class_labels:
        actuals_arr = np.array(actuals)
        class_indices = np.where(actuals_arr == label)[0] 
        class_indices = class_indices.tolist()
        
        class_preds = [predictions[i] for i in class_indices]
        
        class_actuals = [actuals[i] for i in class_indices]
        
        class_acc = accuracy_score(class_actuals, class_preds)
        per_class_accuracy.append(class_acc)

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')

    print(f'{split.capitalize()} Accuracy: {accuracy:.4f}')
    print(f'{split.capitalize()} Precision: {precision:.4f}')
    print(f'{split.capitalize()} Recall: {recall:.4f}')
    print(f'{split.capitalize()} F1-Score: {f1:.4f}')
    print(f'{split.capitalize()} Per-class accuracy (eolian, glacial, beach, fluvial): {*per_class_accuracy,}')

# RESNET-50

model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

predictions_stats(split='train', best_stat='loss')
predictions_stats(split='val', best_stat='loss')
predictions_stats(split='test', best_stat='loss')

