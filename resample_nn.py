# import NumPy library

import numpy as np

# import Decimal library to adjust rounding method

from decimal import Decimal, getcontext, ROUND_HALF_UP

# this function resamples the input image using nearest neighbor interpolation
# and advanced indexing. The two indexing arrays i and j are created to index
# the original image. The 'scale' parameter is a 2 element tuple, where the
# first element indicates the scale factor for the rows of the input image and
# the second element indicates the scale factor for the columns of the input
# image.

def resample_nn(img_in,scale):
    
    # adjust the rounding method to round normally
    
    getcontext().rounding = ROUND_HALF_UP
    
    # get the number of rows in the output image
    
    img_out_row = int(round(Decimal(img_in.shape[0]*scale[0]),0))
    
    img_out_row = np.maximum(img_out_row,1).astype(int)
    
    # get the number of columns in the output image
    
    img_out_col = int(round(Decimal(img_in.shape[1]*scale[1]),0))
    
    img_out_col = np.maximum(img_out_col,1).astype(int)
    
    # generate the advanced indexing array for the input image rows
    
    i = np.linspace(0,img_out_row-1,img_out_row)
    
    i = i / scale[0]
    
    i = np.atleast_2d(i).T
    
    i = np.broadcast_to(i,(img_out_row,img_out_col))
    
    i = np.around(i).astype(np.intp)
    
    i = np.clip(i,None,img_in.shape[0]-1)
    
    # generate the advanced indexing array for the input image columns
    
    j = np.linspace(0,img_out_col-1,img_out_col)
    
    j = j / scale[1]
    
    j = np.broadcast_to(j,(img_out_row,img_out_col))
    
    j = np.around(j).astype(np.intp)
    
    j = np.clip(j,None,img_in.shape[1]-1)
    
    # index the input image using the generated indexing arrays
    
    img_out = img_in[i,j]
        
    return img_out
