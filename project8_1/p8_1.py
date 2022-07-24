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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def findingErrors(compressed, decompressed):
    size = compressed.size
    diff = compressed - decompressed
    rmse = np.sqrt(np.square(diff).sum()/size)
    SNRms = np.square(compressed).sum()/np.square(compressed-decompressed).sum()
    return rmse, SNRms


    
def median_filter(img, center,width, height, kernel):
    img2 = img.copy()
    img2[center[0]-width :center[0]+width  ,
        center[1]-height :center[1]+height ] =\
            cv2.medianBlur(img2[center[0]-width :center[0]+width  ,
        center[1]-height :center[1]+height ] ,kernel)
    return img2
def making_square(img, center, width, height):
    img2 = img.copy()
    img2[center[0]-width :center[0]+width,
         center[1]-height :center[1]+height] = 100
    return img2
# %%
img  = plt.imread('./Fig0801(a).tif')
plt.imshow(img, cmap='gray')
#%%
img_2=median_filter(img, center=[50,27],
                    width=50, height=27, kernel=7)
img_2 = median_filter(img_2, center=[159,30],
                    width=40, height=30, kernel=7)
img_2 = median_filter(img_2, center=[120,90],
                    width=40, height=30, kernel=7)
img_2 = median_filter(img_2, center=[230,10],
                    width=40, height=10, kernel=7)
img_2 = median_filter(img_2, center=[250,25],
                    width=40, height=12, kernel=7)
img_2 = median_filter(img_2, center=[240,150],
                    width=40, height=15, kernel=7)
img_2 = median_filter(img_2, center=[20,220],
                    width=20, height=40, kernel=7)
img_2 = making_square(img_2, [120, 130],4,4)
img_2 = making_square(img_2, [90, 145],4,4)
img_2 = making_square(img_2, [105, 145],4,4)

plt.imshow(img_2, cmap='gray')
# %%
findingErrors(img_2,img)
# %%

img_3 = cv2.filter2D(img, ddepth=-1, kernel=1/4*np.array([[-2,-1,1],
                                             [2,0,4],[1,1,-2]]))
# %%
plt.imshow(img_3, cmap='gray')
# %%
findingErrors(img_3,img)
# %%
im = Image.fromarray(img_3)
im.save("img_b.jpeg")# %%

# %%
im = Image.fromarray(img_2)
im.save("img_c.jpeg")# %%
# %%
