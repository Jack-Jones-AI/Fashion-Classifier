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


# @st.cache(allow_output_mutation=True)
# def load_data(filename=None):
#     filename_default = './data/fashion-mnist_test.csv'
#     if not filename:
#         filename = filename_default

#     df = pd.read_csv(f"./{filename}")
#     return df


# df = load_data()

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

# Download and load the training data
# trainset = datasets.FashionMNIST(
#     './data/fashion-mnist_train.csv', download=True, train=True, transform=transform)
# testset = datasets.FashionMNIST(
#     './data/fashion-mnist_test.csv', download=True, train=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


@st.cache(allow_output_mutation=True)
# Define view_classify function
def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle Boot'], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


###################
# # load the model from disk
PATH = './data/entire_model_v2.pt'
# loaded_model = pickle.load(open(filename, 'rb'))
model = torch.load(PATH)

st.title('Fashion Mninst')
st.write("Zalando")
# print('hello')
st.markdown("""---""")
image = Image.open('./data/zalando_photo_1.jpg')
st.image(image, caption='')


upload = st.file_uploader("Upload image")
if upload is not None:
    st.write(type(upload))
    img = Image.open(upload)
    st.image(img, caption='')
    st.text(" ")
    st.text(" ")

    # # resize
    # new_width = 28
    # new_height = 28
    # img = img.resize((new_width, new_height), Image.ANTIALIAS)
    # # format may what you want *.png, *jpg, *.gif
    # img_res = img.save('./data/output image name.png')

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


# pickle_out = open("model_v1.pkl", mode="wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()


# model_test = model.read_csv('./data/fashion-mnist_test.csv')
# st.write(result)
# st.write("This is a: ", result)

# single_img = model.read_csv(upload)
# max_vals, max_indices = model(single_img).max(1)

# print(max_vals, max_indices)

# pickle_out = open("model_v1.pkl", mode="wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()


# st.markdown("""---""")
# # meet the team
# st.subheader('Jack & Jones Team')

# st.markdown(
#     '[Mark Skinner](https://github.com/aimwps)')
# st.markdown(
#     '[Daniel Biman](https://github.com/DanielBiman)')
# st.markdown(
#     '[Farrukh Bulbulov](https://github.com/fbulbulov)')
# st.markdown(
#     '[Fabio Fistarol](https://github.com/fistadev)')

# st.text(' ')
# st.text(' ')
