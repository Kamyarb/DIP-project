#%%%

import cmath
import math
import timeit
from math import e, log

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy.stats import entropy


def entropycalc(img):
    marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy

img  = plt.imread('./Fig0801(a).tif')
img_b = plt.imread('./img_b.jpeg')
img_c = plt.imread('./img_c.jpeg')
images= [img, img_b, img_c]
# %%
for image in images:
    print(entropycalc(image))    
# %%
