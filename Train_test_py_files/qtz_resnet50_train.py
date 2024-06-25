import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torchvision
from torchvision.transforms import Lambda, Normalize, RandomAdjustSharpness, RandomVerticalFlip, RandomHorizontalFlip, CenterCrop, ToTensor, Resize, ColorJitter, RandomAutocontrast, RandomInvert, RandomRotation, Compose
from torchvision.transforms.functional import rotate
from torchvision.transforms.v2 import RandomEqualize
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms import Lambda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc
import time
from tqdm import tqdm

# Hyperparameters:
num_epochs = 100
learn_rate = 1e-4

'''Class setup: 0 = aeolian, 1 = glacial, 2 = beach, 3 = fluvial'''

device = torch.device("cuda:0")
if torch.cuda.is_available():    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

train_path = r"D:\Michael\Quartz_classifier\data\4_4\train"
val_path = r"D:\Michael\Quartz_classifier\data\4_4\test"

batch_size = 64
img_height = 224
img_width = 224

class PercentileStretchTransform:
    def __init__(self, lower_percentile=5, upper_percentile=95):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self, img):
        # Convert the PIL Image to a PyTorch tensor
        img_tensor = transforms.ToTensor()(img)

        # Convert the tensor to a numpy array
        img_array = img_tensor.numpy()

        # Calculate percentiles
        lower_value = np.percentile(img_array, self.lower_percentile)
        upper_value = np.percentile(img_array, self.upper_percentile)

        # Perform percentile stretching
        stretched_array = np.clip((img_array - lower_value) / (upper_value - lower_value), 0, 1)

        # Convert the numpy array back to a PIL Image
        stretched_img = transforms.ToPILImage()(torch.from_numpy(stretched_array))

        return stretched_img

percentile_stretch_transform = PercentileStretchTransform(lower_percentile=5, upper_percentile=95)

#### Transforms for resnet50
transform = Compose([
    Resize([img_height, img_width]),    
    RandomRotation(degrees=(0,360)),
    RandomAdjustSharpness(sharpness_factor=0.2),
    RandomEqualize(p=1),
    ToTensor(),
])

val_transform = Compose([
    Resize([img_height, img_width]),    
    RandomEqualize(p=1),
    ToTensor(),
])

print('Loading datasets')
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

''' Show examples of the images after transforms have been applied. ''' 
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Generate random indices for image selection
indices = np.random.randint(train_features.size(0), size=(5, 5))

# Create a subplot grid with 5 rows and 5 columns
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# Iterate through the grid and display the images 
for i in range(3):
    for j in range(3):
        num = np.random.randint(batch_size)
        img = train_features[num].squeeze()
        label = train_labels[num]
        axs[i, j].imshow(img.T)
        axs[i, j].set_title(f"Label: {label}")
        axs[i, j].axis('off')

# Adjust spacing between the subplots
plt.tight_layout()
plt.show()

print('Loading pretrained ResNet')
# model = models.resnet101(weights='ResNet101_Weights.DEFAULT', progress=True)
model = models.resnet50(weights='ResNet50_Weights.DEFAULT', progress=True) 

model = model.to(device)

# weight classes based on imbalance
class_sample_count = np.unique(train_dataset.targets, return_counts=True)[1]
class_weights = 1 / torch.tensor(class_sample_count/np.sum(class_sample_count), dtype=torch.float32)
class_weights = class_weights.to(device)

# not being used, left to show how to access class sample counts
# print('class 0 count', class_sample_count[0], 'class 3 count', class_sample_count[3])

num_features = model.fc.in_features
print(num_features)
model.fc = nn.Linear(num_features, 4)
# this is new (1/10) -- added class weights
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learn_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

best_metric = 'loss'
best_metric = 'prc'
best_metric = 'latest'

loss_history = []
accuracy_history = []
precision_history = []
recall_history = []
f1_score_history = []
val_loss_history = []
val_accuracy_history = []
val_precision_history = []
val_recall_history = []
val_f1_history = []

try:
    checkpoint = torch.load(cp_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    # best_prc = checkpoint['best_prc']
    print('Loaded check point model')
except:
    start_epoch = 0
    best_loss = float('inf')
    best_accuracy = float('0')


for epoch in range(start_epoch, num_epochs):
    print("Epoch {} running".format(epoch))

    """ Training Phase """
    start_time = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_samples = 0
    correct_predictions = 0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        # labels = labels.view(-1, 1).float()
        labels = labels.squeeze()
        optimizer.zero_grad()
        model = model.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # '''Debugging things:'''

        # print('loss.item: ', loss.item())
        # print('outputs ', outputs)
        # print('labels ',labels)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        # Calculate metrics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Update counters
        total_samples += labels.size(0)
        correct_predictions += torch.sum(preds == labels)

    print(len(train_dataset))

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    epoch_loss = running_loss / len(train_dataset)
    precision = precision_score(labels_np, preds_np, average='macro')
    recall = recall_score(labels_np, preds_np, average='macro')
    epoch_accuracy = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')

    # Save metrics
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)
    precision_history.append(precision)
    recall_history.append(recall)
    f1_score_history.append(f1)

    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch} completed in: {epoch_duration:.2f} seconds")
    # print(f'[Train #{epoch}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}% Precision: {precision:.4f} Recall: {recall:.4f}%  F1_score: {f1:.4f}')
    print(f'[Train #{epoch}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}% Precision: {precision.item():.4f} Recall: {recall.item():.4f}%  F1_score: {f1.item():.4f}')


    try:
        with open(stats_file, 'a') as file:
            file.write(f"Epoch {epoch} completed in: {epoch_duration:.2f} seconds")
            file.write(f'[Train #{epoch}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}% Precision: {precision:.4f} Recall: {recall:.4f}%  F1_score: {f1:.4f}')
    except:
        pass
    

    """ Validation Phase """
    start_time = time.time()
    model.eval()  # Set model to evaluate mode
    val_running_loss = 0.0

    with torch.no_grad():
        y_true = []
        y_scores = []
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            labels = labels.squeeze()

            outputs = model(inputs)
            loss = criterion(outputs, labels)


            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Calculate metrics
            val_running_loss += loss.item() * inputs.size(0)

            # Update counters
            total_samples += labels.size(0)
            correct_predictions += torch.sum(preds == labels)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_precision = precision_score(labels_np, preds_np, average='macro')
        val_recall = recall_score(labels_np, preds_np, average='macro')
        val_epoch_accuracy = accuracy_score(labels_np, preds_np)
        val_f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='macro')

        scheduler.step(val_epoch_loss)

        # Save metrics
        val_loss_history.append(val_epoch_loss)
        val_accuracy_history.append(val_epoch_accuracy)
        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
        val_f1_history.append(val_f1)
        
        end_time = time.time()
        validation_duration = end_time - start_time
              
        print(f"Validation for Epoch {epoch} completed in: {validation_duration:.2f} seconds")
        print(f'[Val #{epoch}] Loss: {val_epoch_loss:.4f} Acc: {val_epoch_accuracy:.4f}% Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f}')
    
        try:
            with open(stats_file, 'a') as file:
                file.write(f"Validation for Epoch {epoch} completed in: {validation_duration:.2f} seconds")
                file.write(f'[Val #{epoch}] Loss: {val_epoch_loss:.4f} Acc: {val_epoch_accuracy:.4f}% Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f}')
        except:
            pass
    
    if val_epoch_loss < best_loss:
        print(f'Saving model.  Validation loss: {val_epoch_loss:.4f} improved over previous {best_loss:.4f}')
        best_loss = val_epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            # 'best_prc': best_prc,
            'loss_history': loss_history,
            'accuracy_history': accuracy_history,
            'precision_history': precision_history,
            'recall_history': recall_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history,
            'val_precision_history': val_precision_history,
            'val_recall_history': val_recall_history,
            # 'val_prc_history':val_prc_history
            }, "D:/Michael/Quartz_classifier/model/model_checkpoints/4_4/4_4_best_loss.pth")

    if val_epoch_accuracy > best_accuracy:
        print(f'Saving model.  Validation accuracy: {val_epoch_accuracy:.4f} improved over previous {best_accuracy:.4f}')
        best_accuracy = val_epoch_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_acc': best_accuracy,
            'loss_history': loss_history,
            'accuracy_history': accuracy_history,
            'precision_history': precision_history,
            'recall_history': recall_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history,
            'val_precision_history': val_precision_history,
            'val_recall_history': val_recall_history,
            # 'val_prc_history':val_prc_history
            }, "D:/Michael/Quartz_classifier/model/model_checkpoints/4_4/4_4_best_accuracy.pth")

    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'best_loss': best_loss,
    #     # 'best_prc': best_prc,
    #     'loss_history': loss_history,
    #     'accuracy_history': accuracy_history,
    #     'precision_history': precision_history,
    #     'recall_history': recall_history,
    #     'val_loss_history': val_loss_history,
    #     'val_accuracy_history': val_accuracy_history,
    #     'val_precision_history': val_precision_history,
    #     'val_recall_history': val_recall_history,
    #     # 'val_prc_history':val_prc_history
    #     }, "D:/Michael/Quartz_classifier/model/model_checkpoints/qtz_all_modern_3_26_no_braz_oversampled_latest_model.pth")