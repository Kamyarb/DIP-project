#%%%
import cmath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


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



dft = np.fft.fft2(multiply_minus1(img))
img_fft= 50*np.log(np.abs(dft))
inverse_img= multiply_minus1(np.fft.ifft2((dft))).real

plt.figure(figsize=(10,10))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_fft, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(inverse_img, cmap = 'gray')

plt.show()

# %%
print(f' the average value of the image is : {np.mean(img):.2f}' )

print(f' the sum value of the image is : {np.sum(img):.2f}')
print(f' the value of spectrum in center is  : {np.abs(dft[344][344]):.2f}')
# %%
