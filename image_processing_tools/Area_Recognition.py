"""
Process:
    1. process entire image (adaptiveThreshold, contour detection, bounding rect, etc.)
    2. pull boundaries to be convolved
    3. output predictions
"""

import cv2
import pytesseract
import os
import glob
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import math
import imutils

# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\original_bigs")
os.chdir("C:\\Users\\grant\\IS\\IS552\\test")

def longest_line_angle(img_bw):
    """
    Expects image to have white background with black writing
    Function rotates image to various angles,
    iterates every vertical line of pixels in the images,
    line lengths of dark pixels are recorded and the angle returning the longest vertical line is returned

    Warning: slow performance on large images
    """

    angles = [0, -5,-10, -15, -20, -25, -30, -35, -40, 10, 15, 20]
    vert_line_lengths = []
    for indx, angle in enumerate(angles):
        vert_line_lengths.append([angle, 0])
        img_warped = imutils.rotate_bound(img_bw, angle)
        h, w = img_warped.shape[:2]
        for x in range(w):
            line_length = 0
            for y in range(h):
                try:
                    if img_warped[y][x] < 10 and (img_warped[y-1][x] <10 or img_warped[y-1][x-1] <10 or img_warped[y][x-1] <10):
                        line_length += 1
                    else:
                        if line_length > vert_line_lengths[indx][1]:
                            vert_line_lengths[indx][1] = line_length
                        line_length = 0
                except:
                    None

    best_angle_weight = 0
    best_angle = 0
    for indx, val in enumerate(vert_line_lengths):
        if vert_line_lengths[indx][1] > best_angle_weight+1:
            best_angle = vert_line_lengths[indx][0]
            best_angle_weight = vert_line_lengths[indx][1]
    # print("Rotating image to: {}".format(best_angle))
    return best_angle

def show_images(images, cols = 2, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

def add_contours(img_source_bw, img_color, square_pixels):
    image,contours,hierarchy = cv2.findContours(img_source_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img_cont_unfiltered = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_cont_unfiltered,(x,y),(x+w,y+h),(0,255,0),5)
        # First, check for shape of word. Ignoring single letters for now.
        if w > h:
            if (22 <= w <= 500) and (22 <= h <= 200):
                # this ensures that height is less than max_contour, and width, while width is greater than min_contour
                # if (h > min_contour and w < max_contour):
                # """iterate through contours and check for variety in the images"""
                crop_img = img_source_bw[y:y+h, x:x+w]
                # horizontal = columns, vertical = rows
                (horizontal, vertical) = crop_img.shape
                h_lower = int(.2 * horizontal)
                h_upper = int(.8 * horizontal)
                v_lower = int(.2 * vertical)
                v_upper = int(.8 * vertical)

                # sum total changes accross the bw array. If none of middle sixty percentof  arrays contain more than one color change, discard entire boundary
                contained_variety = 0
                for horiz in crop_img[h_lower:h_upper]:
                    row_value_changes = 0
                    value_in_row = horiz[0]
                    for vert_value in horiz[v_lower:v_upper]:
                        if vert_value != value_in_row:
                            row_value_changes += 1
                            value_in_row = vert_value
                    if row_value_changes > 2: # a change and back (we want more than one line)
                        contained_variety += 1
                if contained_variety > 0:
                    cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),5)
                    # file_name = "C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\raw_area_crops\\"+str(v_upper-v_lower+h/w)+"{}.jpg".format(contained_variety*(w/100))
                    # print(x, y, h, w)
                    # raw_output = img_color[y5:y+h, x:x+w] # If really wide, we need larger height for dataset
                    # raw_output = img_color[y-h:y+h+w, x-15:x+w+15]
                    # try:
                    #     raw_output_rotated_expanded = imutils.rotate(raw_output, -longest_line_angle(crop_img))
                    # except:
                    #     raw_output_rotated_expanded = raw_output
                    # cv2.imwrite(file_name, raw_output_rotated_expanded)
                    # print("writing image {}".format(file_name))
                    # cv2.imshow('crop', raw_output_rotated)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
    return img_color, img_cont_unfiltered

for file in glob.glob("*.jpg"):
    print("reading img: ", file)
    img = cv2.imread(file)
    height, width, channels = img.shape
    square_pixels = height * width

    # reduce oversize images
    while square_pixels > 7000000:
        print("reducing oversize image.")
        img = cv2.resize(img, (0,0), fx=0.9, fy=0.9)
        # print("reducing image size...")
        height, width, channels = img.shape
        square_pixels = height * width

    print("creating duplicate images for unrefined output.")
    # unrefined_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # unrefined_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    unrefined_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("creating grayscale image.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(height, width, (square_pixels))

    print("calculating block size for gaussian window:")
    window_block_neighbors = int(.75*math.sqrt(square_pixels))

    while window_block_neighbors %2 != 1:
        window_block_neighbors += 1
    print(window_block_neighbors)

    print("Adding adaptive threshold.")
    # img_low_thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,window_block_neighbors,20)
    # img_low_thresh2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,window_block_neighbors,10)
    img_low_thresh3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,window_block_neighbors,2)

    print("Finding contours.")
    # img_low_thresh,low_thresh_contours,low_thresh_hierarchy = cv2.findContours(img_low_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # img_low_thresh2,low_thresh_contours2,low_thresh_hierarchy2 = cv2.findContours(img_low_thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img_low_thresh3,low_thresh_contours3,low_thresh_hierarchy3 = cv2.findContours(img_low_thresh3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print("Filtering and adding contours.")
    # img_low, img_cont_unfiltered = add_contours(img_low_thresh, unrefined_img)
    # img_low2, img_cont_unfiltered2 = add_contours(img_low_thresh2, unrefined_img2)
    img_low3, img_cont_unfiltered3 = add_contours(img_low_thresh3, unrefined_img3, square_pixels)

    print("Displaying images.")
    # show_images([img_low_thresh, img_low_thresh2, img_low_thresh3, img_cont_unfiltered, img_low])
    # show_images([img_low_thresh, img_low_thresh2, img_low_thresh3, img_low, img_low2, img_low3])
    # show_images([img_low_thresh, img_low_thresh2, img_low, img_low2])
    show_images([img, img_low_thresh3, img_cont_unfiltered3, img_low3])


# TODO split words to characters
# TODO homogenize chracter  sizes

# use machine learning to decide which filters are applicable for contours?
# ignore every contour fully contained within another contour
# everything interesting is generally captured. Try combining overlapping contours when similar height
# try decreasing threshold for bw vs bolding black (adding one to each array)
# rotating rectangles will help ( for rotated records ) - OpenCV has builtin
# ALMOST EVERY SPOT WE WANT TO FIND HAS A HORIZONTAL LINE (sometimes dotted) GOING THROUGH IT with x amount of writing pixels above it, and sometimes some below
# check into repeating patterns to identify dotted line, and slightly shifted solids (fuzzy) for straight line. Bold may help for this task too.
# llook into manual contour creation by checking for curved, connected, lines- can do for just black and white. Probably should decrease threshold rather than thickenning pixels, but try both
# see how open cv does contours
 # could use ratio for contour filter
# next, need to refine boxes to include whole words, where possible.
# maybe combine horizontally similar boxes
# maybe move forward with current results to better understand next steps before spending too much time refining
