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


st.title("Predict image class")
st.markdown("This application can be used to predict "
                " an image")

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

cnames = ['BabyPants', 'BabyShirt', 'womencasualshoes', 'womenchiffontop']
# load pipeline
#new_img = 'BabyShirt_1773.jpg'
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    new_img = Image.open(uploaded_file)
    image = plt.imread(uploaded_file)
    st.image(image)


mm = torch.load('entire_model.pt')
mm.eval()
pred = mm(transform_image(new_img))
probs = torch.nn.functional.softmax(pred, dim=1)
conf, classes = torch.max(probs, 1)


if st.button('Calcola'):
    st.image(image)
    st.write(cnames[classes.item()], 'at confidence score : {0:.2f}'.format(conf.item()))
    
