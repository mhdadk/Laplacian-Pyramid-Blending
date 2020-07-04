# import NumPy library

import numpy as np

# this function accepts a grayscale or RGB image as an input and returns
# an image that is padded according to the 4-element tuple pad_size with
# fill_value. The first element in pad_size determines how many rows
# before the top row of the input image to pad. The second element in pad_size
# determines how many rows after the bottom row of the input image to pad
# The third element in pad_size determines how many columns to pad before the
# first column in the input image. The fourth element in pad_size determines
# how many columns to pad after the last column in the input image.
# For example:
#
# pad_img(img,(1,2,3,4),fill_value=255)
#
# Will pad the input image 'img' with:
#
# - 1 row before filled with 255s
# - 2 rows after filled with 255s
# - 3 columns before filled with 255s
# - 4 columns after filled with 255s

def pad_img(img,pad_size,fill_value=0):
    
    if img.ndim == 2: # grayscale
        
        # initialize new image array filled with fill_value
        
        img_pad = np.full(
                  (img.shape[0]+pad_size[0]+pad_size[1],
                   img.shape[1]+pad_size[2]+pad_size[3]),
                   fill_value,
                   dtype=img.dtype)
        
        # insert the original image into the new image
        
        img_pad[pad_size[0]:img.shape[0]+pad_size[0],
                pad_size[2]:img.shape[1]+pad_size[2]] = img
        
    else: # RGB
        
        # initialize new image array filled with fill_value
        
        img_pad = np.full(
                  (img.shape[0]+pad_size[0]+pad_size[1],
                   img.shape[1]+pad_size[2]+pad_size[3],
                   img.shape[2]),
                   fill_value,
                   dtype=img.dtype)
        
        # insert each channel of the original image into each
        # channel of the new image
        
        for i in range(3):
            
            img_pad[pad_size[0]:img.shape[0]+pad_size[0],
                    pad_size[2]:img.shape[1]+pad_size[2],
                    i] = img[:,:,i]
    
    return img_pad
