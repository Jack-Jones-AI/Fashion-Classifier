import streamlit as st
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


image = Image.open('./data/zalando.jpg')
st.sidebar.image(image, caption=' ')


# image = Image.open('./data/zalando.png')
# st.sidebar.image(image, caption='')

st.sidebar.subheader('Jack & Jones Team')

st.sidebar.markdown(
    '[Mark Skinner](https://github.com/aimwps)')
st.sidebar.markdown(
    '[Daniel Biman](https://github.com/DanielBiman)')
st.sidebar.markdown(
    '[Farrukh Bulbulov](https://github.com/fbulbulov)')
st.sidebar.markdown(
    '[Fabio Fistarol](https://github.com/fistadev)')


###################
# # # load the model from disk
# PATH = './data/entire_model_v2.pt'
# # loaded_model = pickle.load(open(filename, 'rb'))
# model = torch.load(PATH)

st.title('Fashion Mninst')
st.write("Zalando")
# print('hello')
st.markdown("""---""")
image = Image.open('./data/zalando_photo_1.jpg')
st.image(image, caption='')


@st.cache(allow_output_mutation=True)
# load model
class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


PATH = "./data/entire_model_v2.pt"
model = torch.load(PATH)
# model.load_state_dict(torch.load(PATH))
# model.eval()
# st.write(model)

# image -> tensor


def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


####################
upload = st.file_uploader("Upload image")
if upload is not None:
    st.write(type(upload))
    img = Image.open(upload)
    st.image(img, caption='')
    st.text(" ")
    st.text(" ")


# predict


def get_prediction(image_tensor):
    model.eval()
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted, outputs


# predicted, outputs = get_prediction(image_tensor)
# st.write(outputs)

#######


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
n_correct = 0
n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
n_samples += labels.size(0)
n_correct += (predicted == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network on the 10000 test images: {acc} %')


#######


# resize image
# new_width = 28
# new_height = 28
# img = img.resize((new_width, new_height), Image.ANTIALIAS)
# # format may what you want *.png, *jpg, *.gif
# img_res = img.save('./data/resized_image_3.png')

# image = Image.open('./data/resized_image_3.png')
# st.image(image, caption='')


# img_res = Image.open(img_res)
# st.image(img_res, caption='')


# upload.save("converted.png", format="png")
# df2 = pd.read_csv(data_file)


# img = Image.open("converted.png")

# img = img.convert('RGB')

# trans = transforms.ToTensor()

# img = trans(img)

# img.resize_(1, 1, 28, 28)

# result = model.eval()
# st.write(result)
# st.write("This is a: ", model)
