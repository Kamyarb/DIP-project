#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
# %%
def histogram_plot(img):
    plt.hist(img.ravel(),256,[0,256])
# %%
img = plt.imread('./Fig0308(a)(fractured_spine).tif')
# img = plt.imread('./Fig0310(b)(washed_out_pollen_image).tif')

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
# %%
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
# %%
img2 = cdf[img]
# %%
def histogram_equalizer(img):
    equ = cv2.equalizeHist(img)
    
    equalized = np.array(np.power(equ,1.12))
    equalized = np.array(equalized, dtype=np.uint8)
    return equalized

equalized = histogram_equalizer(img)
plt.imshow(equalized, cmap='gray')
res = np.hstack((img,equalized))
plt.figure(figsize=(20,20))

plt.imshow(res, cmap='gray') 
cv2.imwrite('res.png',res)
# %%
plt.hist(equalized)
# %%
plt.hist(img)

# %%
