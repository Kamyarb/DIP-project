#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# %%
def arthimeticOp_add(img1 , img2):
    return cv2.add(img1, img2)

def arthimeticOp_multiply(img1, img2):
        return cv2.multiply(np.array(img1),img2)

def arthimeticOp_divide(img1, img2):
    return cv2.multiply(np.array(img1),img2)

def arthimeticOp_subtract(img1, img2):
    return cv2.subtract(np.array(img1),img2)
# %%
