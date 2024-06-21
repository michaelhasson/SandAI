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

## 3_26 includes Namibia. 3_27 includes Namibia but not Antarctic eolian samples. 
    
# train_path = r"D:\Michael\Quartz_classifier\data\qtz_all_modern_3_26_no_miss_no_braz_oversampled\train"
# val_path = r"D:\Michael\Quartz_classifier\data\qtz_all_modern_3_26_no_miss_no_braz_oversampled\val"
# test_path = r"D:\Michael\Quartz_classifier\data\qtz_all_modern_3_26_no_miss_no_braz_oversampled\test"

train_path = r"D:\Michael\Quartz_classifier\data\4_4_datasets\train_7"
val_path = r"D:\Michael\Quartz_classifier\data\4_4_datasets\val_7"
test_path = r"D:\Michael\Quartz_classifier\data\4_4_datasets\test_7"

# train_path = r"D:\Michael\Quartz_classifier\data\4_4\train"
# val_path = r"D:\Michael\Quartz_classifier\data\4_4\val"
# test_path = r"D:\Michael\Quartz_classifier\data\4_4\test"


img_height = 224
img_width = 224

num_classes = 4

transform = Compose([
    # Lambda(lambda img: img.crop((0, 0, img.width, img.height - 200))),
    Resize((img_height, img_width)),
    # RandomHorizontalFlip(),
    # RandomVerticalFlip(),
    # IncrementalRotate(180),  # up to 180 degrees
    # # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # fine-tune these parameters
    # RandomAutocontrast(0.2),  # consider a probability for this
    # # RandomInvert(0.2),  # consider a probability for this
    RandomEqualize(p=1),
    ToTensor(),
    # Normalize(mean=[0.5], std=[0.5])
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

def predictions_stats(res_num, split, best_stat):
    # res_num: '18', '50'
    # dataset: 'tilewise', 'basinwise'
    # split: 'train', 'val', 'test'
    # best_stat: 'loss', 'prc'
    
    print(f'Predicting with best {best_stat} ResNet-{res_num} on {split} set')
    
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
    # cp_filepath = r"D:\Michael\Quartz_classifier\model\model_checkpoints\keeper_models\qtz_all_modern_11_8_best_loss_model.pth"
    # cp_filepath = r"D:\Michael\Quartz_classifier\model\model_checkpoints\keeper models\qtz_all_modern_3_26_no_braz_oversampled_best_loss_model_1.pth"
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
        # # Save low confidence predictions

        # low_conf_df = pd.DataFrame({'filename': low_confidence_filenames, 'predictions': low_confidence_predictions, 'actuals': low_confidence_actuals})
        # low_conf_df.to_csv(f'D:/Michael/Quartz_classifier/predictions/multiclass_predictions/modern_and_ancient_{split}_low_confidence.csv', index=False)
        # print(f'Saved csv at: D:/Michael/Quartz_classifier/predictions/multiclass_predictions/modern_and_ancient_{split}_low_confidence.csv')
            
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

    # # Print class-wise metrics
    # for class_idx, metrics in classwise_metrics.items():
    #     print(f"Class {class_idx}: Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1-Score {metrics['f1_score']:.4f}")


    #     per_class_accuracy = []
    #     class_labels = [0, 1, 2, 3]

    #     for label in class_labels:
    #         actuals_arr = np.array(actuals) 
    #         class_indices = np.where(actuals_arr == label)[0] 
    #         class_indices = class_indices.tolist()
            
    #         class_preds = [predictions[i] for i in class_indices]
            
    #         class_actuals = [actuals[i] for i in class_indices]
            
    #         class_acc = accuracy_score(class_actuals, class_preds)
    #         per_class_accuracy.append(class_acc)

    #     # accuracy = accuracy_score(filtered_labels, filtered_preds)
    #     # precision = precision_score(filtered_labels, filtered_preds, average='macro')
    #     # recall = recall_score(filtered_labels, filtered_preds, average='macro')
    #     # f1 = f1_score(filtered_labels, filtered_preds, average='macro')

    #     accuracy = accuracy_score(actuals, predictions)
    #     precision = precision_score(actuals, predictions, average='macro')
    #     recall = recall_score(actuals, predictions, average='macro')
    #     f1 = f1_score(actuals, predictions, average='macro')

    #     print(f'{split.capitalize()} Accuracy: {accuracy:.4f}')
    #     print(f'{split.capitalize()} Precision: {precision:.4f}')
    #     print(f'{split.capitalize()} Recall: {recall:.4f}')
    #     print(f'{split.capitalize()} F1-Score: {f1:.4f}')
    #     print(f'{split.capitalize()} Per-class accuracy (aeolian, glacial, beach, fluvial): {*per_class_accuracy,}')
    
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
# model = models.resnet101()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

predictions_stats(res_num=50, split='train', best_stat='loss')
predictions_stats(res_num=50, split='val', best_stat='loss')
predictions_stats(res_num=50, split='test', best_stat='loss')

# # ResNet-101
# model = models.resnet101()
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 4)
# model = model.to(device)

# predictions_stats(res_num=101, split='train', best_stat='loss')
# predictions_stats(res_num=101, split='val', best_stat='loss')
# predictions_stats(res_num=101, split='test', best_stat='loss')

