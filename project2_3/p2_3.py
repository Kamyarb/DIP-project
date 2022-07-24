#%%%

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

image=plt.imread('./Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif')

# %%
k = 10
resized = cv2.resize(image, (image.shape[1]//k,image.shape[0]//k),
                     interpolation = cv2.INTER_NEAREST)


#%%
for m in range(resized.shape[1]):
    for n in range(resized.shape[0]):
        if n==0:
            hstack = resized[n][m]+ np.zeros((10,10))
        else:
            hstack =np.concatenate([hstack , resized[n][m]+np.zeros((10,10))], 
                                   axis=0)
    if m==0:
        vstack = hstack.copy()
    else:
        vstack = np.concatenate([vstack, hstack], axis=1)
# %%

plt.figure(figsize=[30,30])

plt.subplot(1,2,1)
plt.imshow(image , cmap='gray')
plt.title('The real one', fontsize=30)
plt.subplot(1,2,2)
plt.imshow(vstack , cmap='gray')
plt.title('The zoomed one',fontsize=30)
# %%
