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



### TRAINING VARIABLES
LEARNING_RATE = 0.01
EPOCHS = 10
MODEL = JJCNN()
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
ACC_TEST = []
EPOCH_LOSSES = []
RUNNING_LOSS = 0



## TRAINING LOOPS
for epoch in range(EPOCHS):
    running_losses = []
    print(f"EPOCH: {epoch+1}/{epochs}")

    for i, (images, labels) in enumerate(iter(trainloader)):
        #images.resize_(images.size()[0],784)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i%500 ==0:
            print(f"On Iteration: {i}, loss was: {round(running_loss/500, 4)}")
            running_losses.append(running_loss)
            running_loss = 0

    #print(running_losses)
    epoch_losses.append(loss)

    print("\n HERE \n")
    print(epoch_losses)

    #### Validate
    model.eval()
    with torch.no_grad():
        acc = calc_accuracy(model, testloader)
        acc_test.append(acc)
    model.train()
