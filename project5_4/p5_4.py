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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img  = plt.imread('./Fig0526(a)(original_DIP).tif',0)


# %%
T = 1
a = .1
b= .1
width, height = img.shape
H = np.zeros((width, height), dtype='complex')
H[0][0]=1 + 0j
for u in range(1,width):
    for v in range(1,height):
        m = (T/ ((np.pi)*(u*a+v*b)))*\
                (np.sin(np.pi*(u*a+v*b)))*(np.exp((-1j*np.pi*(u*a+v*b))))
        H[u][v] = m

        

# %%
# def multiply_minus1(img):
    
#     df = pd.DataFrame(np.zeros((img.shape[0], img.shape[1])), dtype=int)
#     for i in range(img.shape[0]):
#         if i%2==0:
#             df.loc[i,:] = [ 1 if i%2==0 else -1 for i in range(img.shape[1])]
#         else:
#             df.loc[i,:] = [ -1 if i%2==0 else 1 for i in range(img.shape[1])]
        
#     df = df* img
#     return df.to_numpy()
# # H=1
# # H= multiply_minus1(H)

img  = cv2.imread('./Fig0526(a)(original_DIP).tif',0)
# dft = np.fft.fft2(img)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft = (dft_shift[:,:,0]*1+1j*dft_shift[:,:,1])
# H = np.fft.fftshift(H)
fshift = H*dft
fshift = np.stack([np.real(fshift), np.imag(fshift)], axis=-1)
f_ishift= np.fft.ifftshift(fshift)

img_back = cv2.idft(f_ishift,flags=cv2.DFT_COMPLEX_OUTPUT)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
plt.imshow(img_back, cmap='gray')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


inverse_img = img_back.copy()


plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow((inverse_img),cmap='gray')
plt.subplot(122)
plt.imshow(img,cmap='gray')



# %%
def noisy(noise_typ,image, mean=0 , sigma=1, sprob=0.5,pprop=0.5, amount_sp=0.004):
    if noise_typ == "gauss":
      row,col= image.shape
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col = image.shape
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount_sp * image.size * sprob)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 255

      # Pepper mode
      num_pepper = np.ceil(amount_sp* image.size * (pprop))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
  
noisy_img=noisy('gauss', img_back, mean=0, sigma=10)

# %%
plt.imshow(noisy_img, cmap='gray')
# %%
pspec = (np.abs(dft))**2
noise = 0

wiener = pspec/(pspec+noise)
wiener = wiener*dft
restored = np.fft.ifft2(wiener)

restored = np.real(restored)

# restored = restored.clip(0,255).astype(np.uint8)
# %%
plt.imshow(restored, cmap='gray')
# %%
def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

#TODO : check the wiener filter