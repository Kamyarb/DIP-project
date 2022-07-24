#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage import convolve


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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img = plt.imread('./Fig0338(a)(blurry_moon).tif')
c = 1
m= 3
laplace_kernel= np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
laplacian = convolution2d(img, laplace_kernel,1)
laplacian += min([min(a) for a in laplacian])
img2 = img +np.clip(c* laplacian, 0, 255)
plt.figure(figsize=(10,10))

plt.imshow(img2, cmap='gray')
plt.figure(figsize=(10,10))

plt.imshow(laplacian, cmap='gray')


# %%
