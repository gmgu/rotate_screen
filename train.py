import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Hyper-parameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001


#TODO: load data and split them into train and test datasets
data_X = [] #images
data_Y = [] #labels

train_dataset
test_dataset

classes = (0, 90, 180, 270)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3) # 61, 61
        self.conv2 = nn.Conv2d(6, 16, 4) # 28, 28
        self.conv3 = nn.Conv2d(16, 32, 3) #  12, 12
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2) # 62, 62 -> 31, 31
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2) # 28, 28 -> 14, 14
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2) # 12, 12 -> 6, 6
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for images, labels in enumerate(train_loader):
        #Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # max returns (value ,index)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')