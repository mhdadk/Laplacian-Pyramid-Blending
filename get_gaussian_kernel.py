# import NumPy library

import numpy as np

# import NDimage library

from scipy import ndimage

# this function accepts a 2-element tuple representing window
# size and a standard deviation as inputs and outputs the
# corresponding normalized Gaussian window to be used for
# frequency-domain filtering. For example:
#
# get_gaussian_kernel((512,512),50)
#
# returns a 512 x 512 image consisting of a Gaussian pulse at
# the center with a standard deviation of 50.

def get_gaussian_kernel(kernel_shape,sigma):
    
    # initialize impulse response shape
    
    kernel = np.zeros((kernel_shape[0],kernel_shape[1]))
    
    # compute the indices of the center of the window
    
    kernel_center = (np.floor(kernel_shape[0]/2).astype(int),
                     np.floor(kernel_shape[1]/2).astype(int))
    
    # set impulse at center
    
    kernel[kernel_center] = 1
    
    # compute impulse response
    
    kernel = ndimage.gaussian_filter(kernel,sigma,mode='constant')
    
    # normalize filter such that the center is 1
    
    kernel = kernel / kernel[kernel_center]
    
    return kernel
