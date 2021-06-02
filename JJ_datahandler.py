import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from JJ_model import JJCNN



df_train = pd.read_csv("./Data/fashion-mnist_train.csv")
df_test = pd.read_csv("./Data/fashion-mnist_test.csv")

class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, data, transform = None):
        """Method to initilaize variables."""
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
             # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

#
# loader = transforms.Compose([
#                             transforms.Resize((28,28)),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5), (0.5))
#                               ])
train_set = FashionDataset(df_train, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(df_test, transform=transforms.Compose([transforms.ToTensor()]))

trainloader = DataLoader(train_set, batch_size=100)
testloader = DataLoader(train_set, batch_size=100)
