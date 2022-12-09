#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import requests
import torchvision.transforms as transforms
import torchvision.models as models
import time
import copy
import os

############################################################################################################################################
#Initializing Homepage

st.set_page_config(layout="wide")
st.title('CNN Style Transfer: Create Your Own Digital Art')
st.image('https://th.bing.com/th/id/OIP.v5RJ81B-TZqxVs_avHemRgHaDA?pid=ImgDet&rs=1')#, use_column_width = True)

tab1, tab2 , tab3, tab4= st.tabs(["Content and Style Image Loader (Step 1)", "Fine-Tune Parameters (Step 2)", "Become A Digital Picaso (Step 3)", "Neural Network Metrics (Step 4)"])

############################################################################################################################################
#Loading in data

with tab1: 
    st.subheader('Please insert the URL of a .jpg into the two cells you see below. The first cell is the image you want the style tranform to be applied to and the second cell is the style you want applied.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg19(pretrained=True).to(device).eval()


    imsize = 256
    resize = transforms.Compose([
        transforms.Resize([imsize,imsize]), 
        transforms.ToTensor()])

    def load_image(img_url):
        '''Load image based on url'''
        image = Image.open(requests.get(img_url, stream=True).raw)    
        image = resize(image).unsqueeze(0)
        return image.to(device, torch.float)

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

    content_url = st.text_input('Content Image Link', 'https://www.rcpd.msu.edu/sites/default/files/styles/content_image/public/2021-01/img_5090.jpg?itok=M1x3qSzl', key = 'content')
    style_url   = st.text_input('Style Image Link', 'https://img.xcitefun.net/users/2010/03/155474,xcitefun-digital-art-6.jpg', key = 'style')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('Content Image')
        st.image(content_url)

    with col2:
        st.header('Style Image')
        st.image(style_url)
    
    x_content = load_image(content_url)
    x_style = load_image(style_url)
    
#############################################################################################################################################Fine-Tune Params

with tab2:
    st.header('Below are six seperate options to chose from, they each control various aspects as to how well the style transfer will process.')
    
    st.subheader('Select the Number of iterations that the model uses to learn the Style:')
    num_iterations = st.slider('Number of Iterations', min_value = 10, max_value = 1000, step = 10, value = 100)
    
    st.subheader('Select the weight that the content image carries:')
    content_weight = st.slider('Content Weight', min_value = 0.0, max_value = 2.0, step = 0.1, value = 0.5)
    
    st.subheader('Select the weight that the style image carries:')
    style_weight = st.slider('Style Weight', min_value = 0, max_value = 2000000, step = 10000, value = 1000000)
    
    st.subheader('Select the number of convolutional layers the content image is passed through:')
    content_layers_default = st.multiselect('Select the amount of convolutional layers in this model (Must be ordered):', ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], ['conv_4'])
    
    st.subheader('Select the number of convolutional layers the style image is passed through:')
    style_layers_default = st.multiselect('Select the amount of convolutional layers in this model (Must be ordered):', ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10'], ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']) 
    
    input_img = x_content.clone()
    input_img.requires_grad_(True)
    
    st.subheader('Select the optimizer used in this model:')
    
    tab2a, tab2b, tab2c, tab2d = st.tabs(['SGD','Adam','Adamax','RMSprop'])
    with tab2a:
        st.caption('The SGD or Stochastic Gradient Optimizer is an optimizer in which the weights are updated for each training sample or a small subset of data')
    with tab2b:
        st.caption('Adam Optimizer uses both momentum and adaptive learning rate for better convergence. This is one of the most widely used optimizer for practical purposes for training neural networks.')
    with tab2c:
        st.caption('Adamax optimizer is a variant of Adam optimizer that uses infinity norm. Though it is not used widely in practical work some research shows Adamax results are better than Adam optimizer.')
    with tab2d:
        st.caption('RMSProp applies stochastic gradient with mini-batch and uses adaptive learning rates which means it adjusts the learning rates over time.')    
    
    optim_select = st.selectbox('Choose optimizer', ['SGD','Adam','Adamax','RMSprop'])
    if optim_select == 'SGD':
        optimizer =  optim.SGD([input_img],lr = 0.001, momentum = 0.9)
    elif optim_select == 'Adam':
        optimizer = optim.Adam([input_img],lr = 0.001)
    elif optim_select == 'Adamax':
        optimizer = optim.Adamax([input_img],lr = 0.001)
    elif optim_select == 'RMSprop':
        optimizer =  optim.RMSprop([input_img],lr = 0.001, momentum = 0.9)

#############################################################################################################################################Visualize the network created in previous tab
#with tab3:
    
    
    
    
#############################################################################################################################################Execute the code
#with tab4:
with tab3:
    
    st.header('Below are the parameters you set for your model. Click on the draw button to render your new, stylized image.')
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Number of Iterations", num_iterations)
    col2.metric("Weight of Content", content_weight)
    col3.metric("Weight of Style", style_weight)
    with col4:
        execute = st.button('Draw')
    col5.metric("Number of Content Layers", len(content_layers_default))
    col6.metric("Number of Style Layers", len(style_layers_default))
    col7.metric("Optimizer", optim_select)
    
    col4a, col4b, col4c = st.columns(3)

    with col4a:
        st.subheader('Content Image')
        st.image(content_url, use_column_width = True)

    with col4b:
        st.subheader('Style Image')
        st.image(style_url, use_column_width = True)   
    
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        def forward(self, img):
            return (img - self.mean) / self.std

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class ContentLoss(nn.Module):
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()
        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
        
    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization) 
        i = 0 
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)  
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
    
    if execute == True:
        model, style_losses, content_losses = get_style_model_and_losses(cnn.features,normalization_mean, 
                                                                         normalization_std, x_style, x_content)
        model.requires_grad_(False)
        run = [0]
        content_l = []
        style_l = []
        
        with st.spinner('Painting, give me some quite time!'):
            while run[0] <= num_iterations:
            
                with torch.no_grad():
                    input_img.clamp_(0, 1) 
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                style_l.append(style_score)
                    
                for cl in content_losses:
                    content_score += cl.loss
                content_l.append(content_score)
                    
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                optimizer.step()

        st.success('vualá')
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        with col4c:
            st.subheader('Stylized Image')
            stylized = st.image(torch.permute(torch.squeeze(input_img),(1,2,0)).detach().cpu().numpy(), 
                                use_column_width = True)
            s_data = torch.permute(torch.squeeze(input_img),(1,2,0)).detach().cpu().numpy()
            s_bytes = s_data.tobytes()
            
            btn = st.download_button(label="Download image",
                                     data=s_bytes,
                                     file_name="stylized.jpg")
                
#############################################################################################################################################Visualizing Metrics
with tab4:
    
    st.header('Visualizing your losses')
    
    
    if execute == True:
        cl_y = torch.asarray(content_l).detach().cpu().numpy()
        cl_x = np.arange(0,num_iterations+1,1)
    
        sl_y = torch.asarray(style_l).detach().cpu().numpy()
        sl_x = np.arange(0,len(sl_y),1)
    
        figc, axc = plt.subplots()
        axc.plot(cl_x, cl_y, c = 'orange')
        axc.set_xlabel('# of Iterations')
        axc.set_ylabel('Content Loss Value')
        axc.set_title('Content Loss')
    
        figs, axs = plt.subplots()
        axs.plot(sl_x, sl_y, c = 'green')
        axs.set_yscale('log')
        axs.set_xlabel('# of Iterations')
        axs.set_ylabel('Style Loss Value (log)')
        axs.set_title('Style Loss')
    
        t4col1, t4col2 = st.columns(2)
        with t4col1:
            st.pyplot(figc)
        
        with t4col2:
            st.pyplot(figs)
    
    else:
        st.write('You forgot to draw something!')
    


