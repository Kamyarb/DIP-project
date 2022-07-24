#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img = plt.imread('./Fig0308(a)(fractured_spine).tif')
plt.imshow(img, cmap='gray')
# %%
def log_transform(img , c):
    # if not c:
    # max_level = img.max()
    # c = 255/ ( np.log(1+ max_level))
    # print(c)
    s = c* np.log2(1 + img)
    return np.array(s, dtype = np.uint8)

def power_transform(img, c, gamma):
    return c* np.power(img, gamma)
    
# %%
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(log_transform(img,5), cmap='gray')
plt.subplot(1,3,3)
plt.imshow(power_transform(img,1 ,.5), cmap='gray')


# %%
plt.figure(figsize=(20,20))
i = 1
k = 3
for c in np.linspace(.1,1,k**2):
    # plt.figure(figsize=(20,20))
    plt.subplot(k,k,i)
    plt.imshow(power_transform(img,3 ,c), cmap='gray')
    i+=1
    plt.title(f'for  {c:.2f}')
# %%
