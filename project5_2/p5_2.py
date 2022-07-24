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

def noisy(noise_typ,image, mean=0 , sigma=1, sprob=0.5,pprop=0.5, amount_sp=0.004):
    if noise_typ == "gauss":
      row,col= image.shape
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col = image.shape
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount_sp * image.size * sprob)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 255

      # Pepper mode
      num_pepper = np.ceil(amount_sp* image.size * (pprop))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
# %%
img  = plt.imread('./Fig0507(a)(ckt-board-orig).tif')

img_noisy = noisy('s&p', img, mean=10, sigma=20, amount_sp=0.1, sprob=0.2, pprop=0.2)

# %%
plt.figure(figsize=(15,15))
plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.subplot(122),plt.imshow(img_noisy, cmap = 'gray',)


#%%%%



# %%
img  = plt.imread('./Fig0507(a)(ckt-board-orig).tif')

img_noisy = noisy('s&p', img, mean=10, sigma=20, 
                  amount_sp=0.6, sprob=0.2, 
                  pprop=0.2)

newimg = np.zeros((img_noisy.shape[0], img_noisy.shape[1]))

for i in range(2,img_noisy.shape[0]):
    for j in range(2,img_noisy.shape[1]):
        temp = []
        try:
            temp.append(img_noisy[i-1:i+2, j-1:j+2].ravel())
        except:continue
        temp.sort()
        newimg[i][j] = np.median(temp)
        
# %%
plt.figure(figsize=(15,15))
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.subplot(132),plt.imshow(img_noisy, cmap = 'gray',)
plt.subplot(133),plt.imshow(newimg, cmap = 'gray',)
# %%
