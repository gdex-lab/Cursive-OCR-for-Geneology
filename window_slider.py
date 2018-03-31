"""
Process:
    1. process entire image (adaptiveThreshold, contour detection, bounding rect, rotate, etc.)
    2. Slide and slice based on tone and variety
    3. predict from data set
"""

import os
import glob
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time

os.chdir("C:\\Users\\grant\\IS\\IS552\\JSPapersBookofTheLawoftheLord")


def variety_check(img, w):
    """
    Scan the middle strip of the image for variety of tone
    Converts to black and white.
    Then verifies minimum of 2 changes to color tone
    Intended for small crops of images
    """
    block_size = w
    while block_size %2 != 1:
        block_size -= 1

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # using 10 here to eliminate rows with faint lines
    img_bw = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,block_size,10)
    # cv2.imshow('img' ,img_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    (horizontal, vertical) = img_bw.shape
    h_lower = int(.45 * horizontal)
    h_upper = int(.6 * horizontal)
    v_lower = int(0 * vertical)
    v_upper = int(1 * vertical)

    # print(h_lower, h_upper, v_lower, v_upper)
    # check_area = img_bw[h_lower:h_upper,v_lower:v_upper]
    #
    # cv2.imshow('img' ,check_area)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sum total changes accross the bw array. If none of middle x percentof  arrays contain more than one color change, discard entire boundary
    contained_variety = 0
    for horiz in img_bw[h_lower:h_upper]:
        row_value_changes = 0
        value_in_row = horiz[0]
        for vert_value in horiz[v_lower:v_upper]:
            if vert_value != value_in_row:
                row_value_changes += 1
                value_in_row = vert_value
        if row_value_changes > 2: # a change and back (we want more than one line)
            contained_variety += 1
    if contained_variety > 0:
        # print("Contained variety: ", contained_variety)
        return True
    else:
        return False

def tone_check(crop_rgb_img, h, w, base_tone=100):
    """
    Verify color tone of image passes minimum threshold.
    255 is brightest
    0 is darkest
    Intended for small crops of images
    """
    crop_val = 0
    for line in crop_rgb_img:
        for pixel in line:
            for rgb_val in pixel:
                crop_val += rgb_val
    if crop_val > (h*w*3*base_tone):
        return True
    else:
        return False

def window_slicer(img, w, h, increment_percentage, file_name):
    """
    params:
        img = color image
        w = desired crop width
        h = desired crop heigth
        increment percentage = amount frames will advance from w
        file_name = for output purposes
    """
    w_increment = increment_percentage * w
    h_increment = increment_percentage * h
    img_count = 0

    width_slider = 0
    height_slider = 0
    max_height, max_width = img.shape[:2]

    while height_slider < max_height:
        # crop = img[width_slider:width_slider+h, height_slider:height_slider+w]
        while width_slider < max_width:
            crop = img[height_slider:height_slider+h, width_slider:width_slider+w]
            # print(height_slider, h, width_slider, w)
            if tone_check(crop, h, w) and variety_check(crop, w):
                # cv2.imshow('img', crop)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                out = "C:\\Users\\grant\\IS\\IS552\\JSPapersBookofTheLawoftheLord\\windows_wider\\{}_{}_from_{}".format(height_slider, width_slider, file_name)
                # print("Writing file out {}".format(out))
                # cv2.imwrite(out, crop)
            width_slider += int(w_increment)
        width_slider = 0
        height_slider += int(h_increment)


for file in glob.glob("*.jpg"):
    print("reading img: ", file)
    img = cv2.imread(file)
    window_slicer(img, 70, 60, .3, str(file))
