# import OpenCV library

import cv2 as cv

# this function detects a face in an image and returns the center
# coordinates of that face

def get_face_ctr(img,show_face=False):
    
    # get face classifier
    
    face_classifier = cv.CascadeClassifier(
                      'haarcascade_frontalface_default.xml')
        
    # convert input image to grayscale
    
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # detect face
    
    face_loc = face_classifier.detectMultiScale(gray_img, 1.1, 4)
        
    if len(face_loc): # if a face exists
        
        # compute center coordinates of face
                
        face_ctr = (face_loc[0,1]+(face_loc[0,3]//2),
                    face_loc[0,0]+(face_loc[0,2]//2))
        
        if show_face:
        
            cv.imshow('Detected Face',
                      img[face_loc[0,1]:face_loc[0,1]+face_loc[0,3],
                          face_loc[0,0]:face_loc[0,0]+face_loc[0,2]])
            cv.waitKey()
        
        return face_ctr
    
    else: # if face doesn't exist in image
        
        return 0
