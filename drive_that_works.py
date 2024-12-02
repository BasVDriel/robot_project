# from utils.car_stuff import *

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


def image_preprocessor(image, cell_size):
    """"
    Expects a numpy array as input image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_image.shape
    crop_image = gray_image[h-h//2:, ::]
    resized_image = cv2.resize(crop_image, (w//cell_size, h//cell_size), interpolation=cv2.INTER_LINEAR)
    return resized_image

# Define the model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(256, 128)  # Adjust dimensions according to your input size
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout3(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def hybrid_loss(outputs, targets, class_labels):
    class_labels = torch.tensor(class_labels, dtype=torch.float32, requires_grad=False).detach()
    probs = F.softmax(outputs, dim=1)
    expected_values = torch.sum(probs * class_labels, dim=1).float()
    loss = F.mse_loss(expected_values, targets.float())
    return loss

# Instantiate the model
model = CNNModel(11)

# c = Car.get()
# c.new_image()
# image = c._img
image = cv2.imread('test.jpg')
image = image[:, :, [2, 1, 0]]
cell_size = 20
image = image_preprocessor(image, cell_size)
tensor_image = torch.tensor(image, dtype=torch.float32)

path = 'best_model.pth' 
model.load_state_dict(torch.load(path, weights_only=True))
with torch.no_grad():
      model.eval()  
      tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
      print(tensor_image.shape)
      output =model(tensor_image)
      print(output)


cv2.imshow('image',image)
cv2.waitKey(0)

