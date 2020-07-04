# import the gaussian_blur function

from gaussian_blur import gaussian_blur

# import the resample_nn function

from resample_nn import resample_nn

# import the pad_img function

from pad_img import pad_img

# import OpenCV library

import cv2 as cv

# import NumPy library

import numpy as np

# This function returns the Gaussian and Laplacian pyramids of an image.
# It accepts an RGB or grayscale image as an input. Additionally, it can
# also accept the desired number of layers in the pyramid, including the
# original image, whether to display the pyramids or not, and if
# showing the pyramids, what kind of background to use for the images.
# For example:
#
# (gPyr,lPyr) = get_pyramids(img1,num_layers=3,
#                            show_pyramids=True,
#                            fill_value=255)
#
# This will return the Gaussian pyramid into the gPyr variable, and the
# Laplacian pyramid into the lPyr variable. These pyramids will have 3
# layers, and they will be displayed on a white background.

def get_pyramids(img,num_layers=None,sigma=50,show_pyramids=False,
                 fill_value=0):
    
    # computes the maximum number of layers in the pyramid
    
    max_num_layers = np.around(
                     np.log2(
                     min(img.shape[0],img.shape[1]))).astype(int) - 2
    
    # checks user input for the requested number of layers
            
    if (num_layers == None) or (num_layers > max_num_layers):
        
        num_layers = max_num_layers
    
    # Initialize the lists that will store each layer in the Gaussian
    # and Laplacian pyramids
    
    gaussian_layers = [img]
    
    gaussian_layers_disp = [img]
    
    laplacian_layers = []
    
    laplacian_layers_disp = []
        
    # Downsampling factor
    
    ds_scale = (0.5,0.5)
    
    # Upsampling factor
    
    us_scale = (2,2)
    
    # compute each layer in the Gaussian and Laplacian pyramids
    
    for i in range(num_layers-1):
        
        # blur the current layer to band-limit image for downsampling
        
        blurred_img = gaussian_blur(gaussian_layers[i],sigma)
        
        # downsample blurred layer
        
        ds_img = resample_nn(blurred_img,ds_scale)
        
        # append result to list of gaussian layers
        
        gaussian_layers.append(ds_img)
        
        # pad the resulting image so that it can be displayed later as a
        # pyramid
        
        disp_gaussian_layer = pad_img(ds_img,
        (0,0,0,img.shape[1]-ds_img.shape[1]),
        fill_value=fill_value)
        
        # append padded result to list of gaussian layers to be displayed
        # later
        
        gaussian_layers_disp.append(disp_gaussian_layer)
        
        # upsample the blurred and downsampled image
        
        us_img = resample_nn(ds_img,us_scale)
        
        # blur the upsampled image
        
        blurred_us_img = gaussian_blur(us_img,sigma)
        
        # compute the difference between the result and the current layer
        # of the gaussian pyramid
        
        diff_img = cv.subtract(gaussian_layers[i],blurred_us_img)
        
        # append the result to the list of laplacian layers
        
        laplacian_layers.append(diff_img)
        
        # pad the result to display the laplacian pyramid later
        
        disp_laplacian_layer = pad_img(diff_img,
        (0,0,0,img.shape[1]-diff_img.shape[1]),
        fill_value=fill_value)
        
        # append it to the list of padded laplacian layers
        
        laplacian_layers_disp.append(disp_laplacian_layer)
    
    # append the last gaussian layer to the list of laplacian layers
    
    laplacian_layers.append(gaussian_layers[-1])
    
    # repeat for the padded images
    
    laplacian_layers_disp.append(gaussian_layers_disp[-1])
    
    if show_pyramids:
        
        # display gaussian pyramid
        
        cv.namedWindow('Gaussian Pyramid',cv.WINDOW_NORMAL)
        
        cv.imshow('Gaussian Pyramid',np.vstack(gaussian_layers_disp))
        
        # display laplacian pyramid
        
        cv.namedWindow('Laplacian Pyramid',cv.WINDOW_NORMAL)
        
        cv.imshow('Laplacian Pyramid',np.vstack(laplacian_layers_disp))
        
    return (gaussian_layers,laplacian_layers)
  