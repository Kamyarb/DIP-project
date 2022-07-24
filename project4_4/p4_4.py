#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cmath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import math

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img  = plt.imread('./Fig0441(a)(characters_test_pattern).tif')

def multiply_minus1(img):
    
    df = pd.DataFrame(np.zeros((img.shape[0], img.shape[1])), dtype=int)
    for i in range(img.shape[0]):
        if i%2==0:
            df.loc[i,:] = [ 1 if i%2==0 else -1 for i in range(img.shape[1])]
        else:
            df.loc[i,:] = [ -1 if i%2==0 else 1 for i in range(img.shape[1])]
        
    df = df* img
    return df.to_numpy()

def get_gauss_kernel(size=[4,4],sigma=1, center=None):
    if center is None:
        center=(size[0]//2 ,size[1]//2)
    kernel=np.zeros((size[0],size[1]))
    for i in range(size[0]):
       for j in range(size[1]):
          diff=np.sqrt((i-center[0])**2+(j-center[1])**2)
          kernel[i,j]=np.exp(-((0.4*diff)**2)/(2*sigma**2))
    return kernel/np.sum(kernel)

def GaussHPF(size=[4,4],sigma=1, center=None):
    if center is None:
        center=(size[0]//2 ,size[1]//2)
    kernel=np.zeros((size[0],size[1]))
    for i in range(size[0]):
       for j in range(size[1]):
          diff=np.sqrt((i-center[0])**2+(j-center[1])**2)
          kernel[i,j]= 1 - np.exp(-((0.5*diff)**2)/(2*sigma**2))
    return kernel/np.sum(kernel)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dft = np.fft.fft2(multiply_minus1(img))
gaussianHPF= GaussHPF([img.shape[0], img.shape[1]] , sigma=20)
filterdspectrum = gaussianHPF * dft
inverse_img= multiply_minus1(np.fft.ifft2((filterdspectrum))).real

plt.figure(figsize=(15,15))
plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(gaussianHPF, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow((inverse_img).clip(0,255), cmap = 'gray')
# %%
