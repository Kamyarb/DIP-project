#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cmath
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

#%%%
img  = plt.imread('./Fig0526(a)(original_DIP).tif')
# %%%%%%%%%%% a %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def add_sine_noise(img, A,u0,v0):
    width, height = img.shape
    noisy_img = img.copy()
    for i in range(width):
        for j in range(height):
            noisy_img[i][j] += A * np.sin(v0 * i + u0*j )
    return noisy_img
            
# %% b %%%%%%%%%%%

noisy_img  = add_sine_noise(img, 40, 0, img.shape[0]//2)
plt.figure(figsize=(15,15))
plt.subplot(131),plt.imshow(img, cmap = 'gray')

plt.subplot(132),plt.imshow(noisy_img, cmap = 'gray')
# plt.subplot(133),plt.imshow(add_sine_noise(img, 40,0,img.shape[1]//2), cmap = 'gray',)

# %%
def multiply_minus1(img):
    
    df = pd.DataFrame(np.zeros((img.shape[0], img.shape[1])), dtype=int)
    for i in range(img.shape[0]):
        if i%2==0:
            df.loc[i,:] = [ 1 if i%2==0 else -1 for i in range(img.shape[1])]
        else:
            df.loc[i,:] = [ -1 if i%2==0 else 1 for i in range(img.shape[1])]
        
    df = df* img
    return df.to_numpy()


dft = np.fft.fft2(multiply_minus1(img))
img_fft= 20*np.log(np.abs(dft))

dft_noisy = np.fft.fft2(multiply_minus1(noisy_img))
img_fft_noisy= 20*np.log(np.abs(dft_noisy))


# inverse_img= multiply_minus1(np.fft.ifft2((dft))).real
# %%
plt.figure(figsize=(15,15))
plt.subplot(131),plt.imshow(noisy_img, cmap = 'gray')
plt.title('original image')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_fft, cmap = 'gray')
plt.title('spectrum of original image')

# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_fft_noisy, cmap = 'gray')
plt.title('spectrum of noisy image')

# %%

def notch_filter(dft, a):
    # dft2 = dft.copy()
    # k = 20
    # dft2[:,:344-k] = 0
    # dft2[:,344+k:] = 0
    # dft2[344-a: 344+a , 340:350]=dft[344-a: 344+a , 340:350].copy()
    mask = np.zeros(dft.shape)
    mask[:, 344:347] = 1
    mask[344-a: 344+a , 344:347] = 0
    dft2 = dft * mask
    return dft2
    
# %%
dft_notch = notch_filter(dft_noisy, a=1)
inverse_img= multiply_minus1(np.fft.ifft2((dft_notch))).real
# %%
plt.imshow(inverse_img, cmap = 'gray')
# %%
plt.imshow(20*np.log(np.abs(dft_notch)+1), cmap='gray')
# %%
