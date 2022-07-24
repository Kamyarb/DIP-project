#%%%

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
image=plt.imread('Fig0221(a)(ctskull-256).tif')

# %%
def changeLevels(desiredLevels):
    k = np.log2(desiredLevels)
    intensity_level = 2**(8-k)
    target_compr_factor = 256/intensity_level
    image_reduced = np.floor(image/256 * target_compr_factor)
    return image_reduced



# %%
plt.figure(figsize=[30,30])
for k in [2**i for i in range(1,9)]:

    plt.subplot(3,3 ,int(np.log2(k)))
    plt.imshow(changeLevels(k), cmap='gray')
    plt.title('Grey-level  '+str(k), fontsize=30)
# %%
