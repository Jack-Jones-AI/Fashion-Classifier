import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from invert import Invert
import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from PIL import Image

MODEL = torch.load("trained_models/trained_model.pt")



def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()
    return ax


def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


IMG_DIR = './img_test/boot.jpeg'
loader = transforms.Compose([
                            # Invert(),
                            transforms.ColorJitter(brightness=(0.5,1.5), contrast=(10), saturation=(0.5,1.5) ),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize((28,28)),
                            transforms.ToTensor(),])
                            #transforms.Normalize((0.5), (0.5))
                              #])
#hue=(-0.1,0.1)
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    print(f"BEFORE UNSQUEEZE {image.shape}")

    image = image.unsqueeze(0)
    return image

ti = image_loader(IMG_DIR)
# imshow(ti.squeeze(0), cmap="gray")
imshow(ti.squeeze(0))
MODEL.eval()
output = MODEL(ti)
tensor, label = torch.max(output, 1)
print(label)
print(output)
print(output_label(label.item()))
