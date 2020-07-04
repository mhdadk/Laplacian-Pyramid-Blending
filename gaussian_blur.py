# import the pad_img() function

from pad_img import pad_img

# import the get_gaussian_kernel() function

from get_gaussian_kernel import get_gaussian_kernel

# import NumPy library

import numpy as np

# import FFT_2D function

from FFT_2D import FFT_2D

# import iFFT_2D function

from iFFT_2D import iFFT_2D

# import fftpack library

from scipy import fftpack as dsp

# this function performs gaussian blurring of an image.
# It accepts an image as an input and an optional standard deviation.

def gaussian_blur(img,sigma=50):
    
    # define padding size as twice the dimensions of the original image
    # to ensure linear, and not circular, convolution in spatial domain
    
    pad_size = (np.floor(img.shape[0]/2).astype(int),
                np.ceil(img.shape[0]/2).astype(int),
                np.floor(img.shape[1]/2).astype(int),
                np.ceil(img.shape[1]/2).astype(int))
    
    # pad input image with zeros using the pad_img function
    
    padded_img = pad_img(img,pad_size)
    
    # generate the gaussian frequency window to be used for
    # gaussian blur
    
    freq_window = get_gaussian_kernel((padded_img.shape[0],
                                       padded_img.shape[1]),
                                       sigma)
    
    if img.ndim == 2: # grayscale
        
        # perform 2D FFT using hand-made function
        
        padded_img_FFT = FFT_2D(padded_img)
        
        # shift the FFT to center of frequency spectrum
        
        padded_img_FFT = dsp.fftshift(padded_img_FFT)
        
        # this is the same as spatial convolution
        
        G = np.multiply(padded_img_FFT,freq_window)
        
        # shift result back to zero frequency
        
        G = dsp.ifftshift(G)
        
        # perform inverse 2D FFT using hand-made function and then
        # constrain values to the range 0 - 255
        
        g = np.clip(np.abs(iFFT_2D(G)),0,255).astype(np.uint8)
        
        # extract filtered image from zero-padded image
        
        img_out = g[np.floor(padded_img.shape[0]/4).astype(int):
                    np.floor(padded_img.shape[0]/4).astype(int)+
                    np.floor(padded_img.shape[0]/2).astype(int),
                    np.floor(padded_img.shape[1]/4).astype(int):
                    np.floor(padded_img.shape[1]/4).astype(int)+
                    np.floor(padded_img.shape[1]/2).astype(int)]
    
    else: # RGB
        
        # initialize 3-channel arrays to store results
        
        padded_img_FFT = np.zeros(padded_img.shape,dtype=np.complex128)
        
        G = np.zeros(padded_img.shape,dtype=np.complex128)
        
        g = np.zeros_like(padded_img)
        
        img_out = np.zeros_like(img)
        
        # loop for each color channel
        
        for i in range(3):
            
            # perform 2D FFT using hand-made function
            
            padded_img_FFT[:,:,i] = FFT_2D(padded_img[:,:,i])
            
            # shift the FFT to center of frequency spectrum 
            
            padded_img_FFT[:,:,i] = dsp.fftshift(padded_img_FFT[:,:,i])
            
            # this is the same as spatial convolution
            
            G[:,:,i] = np.multiply(padded_img_FFT[:,:,i],freq_window)
            
            # shift result back to zero frequency
            
            G[:,:,i] = dsp.ifftshift(G[:,:,i])
            
            # perform inverse 2D FFT using hand-made function and then
            # constrain values to the range 0 - 255
            
            g[:,:,i] = np.clip(np.abs(
                               iFFT_2D(G[:,:,i])),0,255).astype(np.uint8)
            
            # extract filtered channel from zero-padded channel
            
            img_out[:,:,i] = g[np.floor(padded_img.shape[0]/4).astype(int):
                               np.floor(padded_img.shape[0]/4).astype(int)+
                               np.floor(padded_img.shape[0]/2).astype(int),
                               np.floor(padded_img.shape[1]/4).astype(int):
                               np.floor(padded_img.shape[1]/4).astype(int)+
                               np.floor(padded_img.shape[1]/2).astype(int),
                               i]
    return img_out
