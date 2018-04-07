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
import statistics
import imutils

os.chdir("C:\\Users\\grant\\IS\\IS552\\JSPapersBookofTheLawoftheLord")

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

    return best_angle

def show_images(images, cols = 2, titles = None):
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
    average_angle = 0
    angles = []
    applied_contour_count = 0
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
                    applied_contour_count += 1
                    cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),5)

                    if applied_contour_count % 5 == 0:
                        angles.append(longest_line_angle(crop_img))
                    # file_name = "C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\raw_area_crops\\"+str(v_upper-v_lower+h/w)+"{}.jpg".format(contained_variety*(w/100))
                    # print(x, y, h, w)
                    # raw_output = img_color[y5:y+h, x:x+w] # If really wide, we need larger height for dataset
                    # raw_output = img_color[y-h:y+h+w, x-15:x+w+15]
                    # try:
                    #     raw_output_rotated_expanded = imutils.rotate(raw_output, -longest_line_angle(crop_img))
                    # except:
                    #     raw_output_rotated_expanded = raw_output

    angles.sort()
    average_angle = statistics.median(angles)

    return img_color, img_cont_unfiltered, average_angle

for file in glob.glob("*.jpg"):
    print("reading img: ", file)
    img = cv2.imread(file)

    print("creating duplicate images for immutable output.")
    original_img = img.copy()
    height, width, channels = img.shape
    square_pixels = height * width

    # while square_pixels > 7000000:
    #     print("reducing oversize image: {}".format(square_pixels))
    #     img = cv2.resize(img, (0,0), fx=0.9, fy=0.9)
    #     height, width, channels = img.shape
    #     square_pixels = height * width

    print("Dilating image to repair errosion")
    kernel = np.ones((5,9),np.uint8) # was 5, 5, first is vert, then horizontal
    img_dilated = cv2.erode(img,kernel,iterations = 1)
    # erosion extracts the white from the image and replaces with surrounding dark.
    # In this case, the writing is dark, and therefore the erosion works as dilation
    # dilation = cv2.dilate(img,kernel,iterations = 1)
    # show_images([img, erosion, dilation])


    print("creating grayscale image.")
    img_gray = cv2.cvtColor(img_dilated, cv2.COLOR_BGR2GRAY)
    print(height, width, (square_pixels))


    print("calculating block size for gaussian window:")
    # The smaller this block size, the more area-sensitive the window will be
    #.75 isn't capturing all we need (but that was before -20 thresh)
    window_block_neighbors = int(.75*math.sqrt(square_pixels))

    while window_block_neighbors %2 != 1:
        window_block_neighbors += 1


    print("Adding adaptive threshold.")
    # The last param is a manual input which subtracts from the threshold
    img_low_thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,window_block_neighbors,-10)

    print("Finding contours.")
    img_low_thresh,low_thresh_contours,low_thresh_hierarchy = cv2.findContours(img_low_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print("Filtering and adding contours.")
    img_low, img_cont_unfiltered, angle = add_contours(img_low_thresh, img, square_pixels)

    # print("Rotating image to match writing angle")
    # For creation of data set
    # rotated_img = imutils.rotate_bound(original_img, angle)
    # file_name = "C:\\Users\\grant\\IS\\IS552\\JSPapersBookofTheLawoftheLord\\RotationsApplied\\{}".format(file)
    # print("Writing rotated image to file: {}".format(file_name))
    # cv2.imwrite(file_name, rotated_img)

    print("Displaying images.")
    show_images([img, img_low_thresh, img_cont_unfiltered, img_low])
