# import the get_pyramids function

from get_pyramids import get_pyramids

# import the resample_nn function

from resample_nn import resample_nn

# import the gaussian_blur function

from gaussian_blur import gaussian_blur

from get_face_ctr import get_face_ctr

# import the OpenCV library

import cv2 as cv

# import the NumPy library

import numpy as np

# This function performs Laplacian blending on two images. It accepts two
# N x N images. It also accepts other inputs. For example:
#
# laplacian_blend(img1,img2,ROI,sigma=40,show_blend=True)
#
# This returns an image that is blended with a standard deviation of 40
# and the blended image is displayed.

def laplacian_blend(img1,img2,ROI,sigma=20,show_blend=False):
    
    (x,y,w,h) = ROI
    
    # get the laplacian pyramid for the first image
    
    LA = get_pyramids(img1,sigma=sigma,show_pyramids=True)[1]
    
    # get the laplacian pyramid for the second image
    
    LB = get_pyramids(img2,sigma=sigma,show_pyramids=True)[1]
    
    face_ctr = get_face_ctr(img2)
    
    # create a binary mask for the image based on the selected ROI    
        
    binary_mask = np.copy(img2)
    
    binary_mask[:,:,:] = 0
    
    # align binary mask with face
    
    if face_ctr != 0:
        
        binary_mask[face_ctr[0]-(h//2):face_ctr[0]+(h//2),
                    face_ctr[1]-(w//2):face_ctr[1]+(w//2)] = 255
    
    else:
        
        binary_mask[y:y+h,x:x+w] = 255
    
    # get the gaussian pyramid for the binary mask
    
    GR = get_pyramids(binary_mask,sigma=sigma,show_pyramids=True)[0]
    
    # initialize the list to store the combined laplacian pyramid
    
    LS = []
    
    for la,lb,gr in zip(LA,LB,GR):
        
        # normalize binary masks at each layer to use as weights
        
        gr = gr / 255
        
        # get inverse of normalized binary mask at each layer
        
        gr_inv = 1.0 - gr
        
        # compute the combined laplacian layer
        
        ls = np.multiply(gr,la) + np.multiply(gr_inv,lb)
        
        # append result to the list of combined laplacian layers
        
        LS.append(ls.astype(np.uint8))
    
    # reverse the list of combined laplacian layers for reconstruction        
    
    LS.reverse()
    
    # start image reconstruction from the bottom up
        
    blended_img = LS[0]
        
    # upsampling factor
        
    us_scale = (2,2)
    
    for i in range(1,len(LS)):
        
        # upsample the layer
        
        blended_img = resample_nn(blended_img,us_scale)
        
        # blur it
        
        blended_img = gaussian_blur(blended_img,sigma)
        
        # add it to the above layer to get to the next layer
        
        blended_img = cv.add(blended_img, LS[i])
    
    if show_blend:
        
        # display blended image
        
        cv.namedWindow('Blended Image',cv.WINDOW_NORMAL)
        
        cv.imshow('Blended Image',blended_img)
    
    return blended_img
