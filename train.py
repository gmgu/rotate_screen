import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
import PIL
import time
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

PATH = './trained_model/cnn.pth'

# torch.manual_seed(0)

# Hyper-parameters
num_epochs = 4
batch_size = 8
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)


##################################################################################
# DataSet
class ImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = PIL.Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


classes = (0, 90, 270)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.FiveCrop((224, 224)),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Normalize(0.5, 0.5)
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


target_transform = transforms.Lambda(lambda y: torch.tensor(int(0 if y == 0 else (1 if y == 90 else 2))))

train_dataset = ImageDataset("train_data/annotation.csv", "train_data",
                             transform=transform, target_transform=target_transform)
test_dataset = ImageDataset("test_data/annotation.csv", "test_data",
                            transform=test_transform, target_transform=target_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


images, labels = next(iter(train_loader))
print(f"Feature batch shape: {images.size()}")
print(f"Labels batch shape: {labels.size()}")


##################################################################################
# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)  # 222, 222 -> 111, 111
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)  # 109, 109 -> 54, 54
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)  # 52, 52 -> 26, 26
        x = F.max_pool2d(F.relu(self.conv4(x)), 2, 2)  # 24, 24 -> 12, 12

        # print(x.size())
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = CNN().to(device)
model.load_state_dict(torch.load(PATH))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("total batch: {}, total epoch: {}".format(len(train_loader), num_epochs))

start = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        images = images.to(device)
        labels = labels.to(device)

        #Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)  # max returns (value ,index)
        running_acc += torch.sum(predicted == labels)/labels.size(0)
        running_loss += loss.item()
        if i % 100 == 99:
            print("Epoch: {}, Iter: {}, Loss: {:.2f} Accuracy: {:.2f} %".format(epoch + 1, i + 1, running_loss, running_acc))
            running_acc = 0
            running_loss = 0
elapsed_time = time.time() - start

print('Finished Training: {:.0f} min {:.0f} sec'.format(elapsed_time // 60, elapsed_time % 60))
print('Saving model...', end='')
torch.save(model.state_dict(), PATH)
print('Done.')

##########################################################test
print('Testing...', end='')
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(3)]
    n_class_samples = [0 for i in range(3)]
    for iter, (images, labels) in enumerate(test_loader):  # five crop - images is 5d tensor
        bs, ncrops, c, h, w = images.size()
        images = images.view(-1, c, h, w)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        outputs = outputs.view(bs, ncrops, -1).mean(1)

        _, predicted = torch.max(outputs, 1)  # max returns (value ,index)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Test Accuracy: {acc:.2f} %')

    for i in range(3):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f} %')
