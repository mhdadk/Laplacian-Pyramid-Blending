# import OpenCV library

import cv2 as cv

# import Tk and messagebox libraries from tkinter package

from tkinter import Tk,messagebox

# import askopenfilename() function

from tkinter.filedialog import askopenfilename

# import laplacian_blend function

from laplacian_blend import laplacian_blend

#----------------------------------------------------------------------
# ask user to open first image

Tk().withdraw()

instructions_str_1 = """Please open the image that you want
to extract a region of interest from."""

messagebox.showinfo("First Step",instructions_str_1)

#----------------------------------------------------------------------
# Open import window

Tk().withdraw()

filename1 = askopenfilename() 

#----------------------------------------------------------------------
# ask user to open second image

Tk().withdraw()

instructions_str_2 = """Now open the image that you want to
blend the region of interest with."""

messagebox.showinfo("Second Step",instructions_str_2)

#----------------------------------------------------------------------
# Open import window again

Tk().withdraw()

filename2 = askopenfilename() 

#----------------------------------------------------------------------
# read both images

img1 = cv.imread(filename1)

img2 = cv.imread(filename2)

#----------------------------------------------------------------------
# ask user to select region of interest

Tk().withdraw()

instructions_str_3 = """Finally, select the rectangular region of interest
by left-clicking your mouse and dragging."""

messagebox.showinfo("Final Step",instructions_str_3)

#----------------------------------------------------------------------
# get region of interest

ROI = cv.selectROI(img1)

cv.destroyAllWindows()

#----------------------------------------------------------------------
# perform Laplacian blending of images

laplacian_blend(img1,img2,ROI,sigma=30,show_blend=True)