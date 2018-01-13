import cv2
import pytesseract
import os
import glob
import cv2
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image


# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\Isolated")
os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\1900s")
# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\Isolated\\BlackandWhite\\Rotated")

def countours(img):
    '''
    image must be gray or bw
    '''
    # Find the contours
    ret,thresh = cv2.threshold(img,127,255,0)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
    return img

def auto_bw_plus(img):
    '''
    converts image to black and white with adaptive Gaussian threshold for
     images with varying lighting
    Conversion does not perserve or enhace writing, since CNN will process all
     attributes. Simply working towards clearer text area identification
    '''

    # smooth the image to avoid noises
    img = cv2.medianBlur(img,1)

    # Apply adaptive threshold
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    return img

for file in glob.glob("*.jpg"):
    # original
    img = cv2.imread(file, 0)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # black and white from gaussian threshold
    cv2.imshow('img', auto_bw_plus(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # countours
    cv2.imshow('img', countours(auto_bw_plus(img)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
