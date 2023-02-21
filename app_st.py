import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import datasets, transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transform
    #image = Image.open(image_bytes)
    return my_transforms(image_bytes).unsqueeze(0)


st.title("Deep learning to predict the class of an image")
st.markdown("This application can be used to predict "
                " the class of a clothing item")
st.write('''The data is a subset from the Shopee-IET dataset from Kaggle. 
           Each image in this data depicts a clothing item and 
           the corresponding label specifies its clothing category.
        The following possible labels: BabyPants, BabyShirt, womencasualshoes, womenchiffontop.''')

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

cnames = ['BabyPants', 'BabyShirt', 'womencasualshoes', 'womenchiffontop']
# load pipeline
# img = 'BabyShirt_1773.jpg'

def select_img(img):
    d = {'Image 1': 'BabyShirt_1773.jpg',
        'Image 2': 'womencasualshoes_2240.jpg'
        }
    return d[img]     

## Select the filename
img_file = st.sidebar.selectbox("Choose an image", ("Image 1", "Image 2"))

        
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    new_img = Image.open(uploaded_file)
    image = plt.imread(uploaded_file)
    st.write('You uploaded this image')
    st.image(image)
else:
    img = select_img(img_file)
    new_img = Image.open(img)
    image = plt.imread(img)
    st.write('Your test image')
    st.image(image)         


mm = torch.load('entire_model.pt')
mm.eval()
pred = mm(transform_image(new_img))
probs = torch.nn.functional.softmax(pred, dim=1)
conf, classes = torch.max(probs, 1)


if st.button('Calcola'):
    st.image(image)
    st.write(cnames[classes.item()], 'at confidence score : {0:.2f}'.format(conf.item()))
    

st.sidebar.markdown('''
    ----------
    By Danielle Taneyo, PhD''')