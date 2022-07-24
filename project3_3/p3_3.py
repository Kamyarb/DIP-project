#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage.filters import convolve

#%%%%%%


def convolution2d(image, kernel, pad):
    m, n = kernel.shape
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    y, x = image.shape
    y_out = y - m + 1
    x_out  = x - n + 1
    new_image = np.zeros((y_out, x_out))
    for i in range(y_out):
        for j in range(x_out):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n]*kernel)
    return new_image



#%%%%%%

 

def conv2d(image, kernel):
    im = convolve(image, kernel)
    im = np.array(np.clip(im, 0, 255), dtype=np.uint8) 
    return im


# %%
img = plt.imread('./Fig0308(a)(fractured_spine).tif')
# %%
m=7
img_2=conv2d(img, 1/m**2 *np.ones((m,m)))
# %%
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_2, cmap='gray')
# %%
