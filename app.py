import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
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


PATH = "./data/state_dict_model2.pt"
model = torch.load(PATH)
# PATH = "./data/entire_model_v2.pt"
# model.load_state_dict(torch.load(PATH))
# model.eval()
# st.write(model)

# Define view_classify function


# def view_classify(img, ps, version="MNIST"):
#     ''' Function for viewing an image and it's predicted classes.
#     '''
#     ps = ps.data.numpy().squeeze()

#     fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
#     ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
#     ax1.axis('off')
#     ax2.barh(np.arange(10), ps)
#     ax2.set_aspect(0.1)
#     ax2.set_yticks(np.arange(10))
#     if version == "MNIST":
#         ax2.set_yticklabels(np.arange(10))
#     elif version == "Fashion":
#         ax2.set_yticklabels(['T-shirt/top',
#                              'Trouser',
#                              'Pullover',
#                              'Dress',
#                              'Coat',
#                              'Sandal',
#                              'Shirt',
#                              'Sneaker',
#                              'Bag',
#                              'Ankle Boot'], size='small')
#     ax2.set_title('Class Probability')
#     ax2.set_xlim(0, 1.1)

#     plt.tight_layout()
#     st.pyplot(fig)

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


# image -> tensor


# def transform_image(image_bytes):
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# image = Image.open(io.BytesIO(image_bytes))
# return transform(image).unsqueeze(0)


def image_loader(image_name):
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

# st.write(type(image))

# predict


def get_prediction(image_tensor):
    # model.eval()
    images = image_tensor
    # images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted, outputs


####################
upload = st.file_uploader("Upload image")
if upload is not None:
    st.write(type(upload))
    img = Image.open(upload)
    st.image(img, caption='')
    transf = image_loader(upload)
    outputs = model(transf)
    _, predicted = torch.max(outputs, 1)
    # print(type(transf))
    st.write(type(transf))
    # predicted = get_prediction(transf)
    st.write(outputs)

    st.text(" ")
    st.text(" ")


#######


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
