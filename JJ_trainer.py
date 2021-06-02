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


def calc_accuracy(mdl, test_data):
    # reduce/collapse the classification dimension according to max op
    # resulting in most likely label
    total_acc = []
    for images, labels in iter(test_data):
        #images.resize_(images.size()[0],784)
        max_vals, max_indices = mdl(images).max(1)
        # assumes the first dimension is batch size
        n = max_indices.size(0)  # index 0 for extracting the # of elements
        # calulate acc (note .item() to do float division)
        acc = (max_indices == labels).sum().item() / n
        total_acc.append(acc)

    final_acc = sum(total_acc) / len(total_acc)
    print(f"The average accuracy across all tests: {final_acc}, test_size: {len(total_acc)}")
    return final_acc


### TRAINING VARIABLES
LEARNING_RATE = 0.01
EPOCHS = 10
MODEL = JJCNN()
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
ACC_TEST = []
EPOCH_LOSSES = []
RUNNING_LOSS = 0
MODEL_SAVE_PATH = "./trained_models"


## TRAINING LOOPS
for epoch in range(EPOCHS):
    running_losses = []
    print(f"EPOCH: {epoch+1}/{EPOCHS}")

    for i, (images, labels) in enumerate(iter(trainloader)):
        #images.resize_(images.size()[0],784)
        OPTIMIZER.zero_grad()
        output = MODEL.forward(images)
        loss = CRITERION(output, labels)
        loss.backward()
        OPTIMIZER.step()
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
    MODEL.eval()
    with torch.no_grad():
        acc = calc_accuracy(MODEL, testloader)
        acc_test.append(acc)
    MODEL.train()


torch.save(MODEL, MODEL_SAVE_PATH)
