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
# print(pytesseract.pytesseract.tesseract_cmd)
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\grant\\AppData\\Local\\Tesseract-OCR'
# print(pytesseract.pytesseract.tesseract_cmd)
# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\Isolated")
# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\1900s")
os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc")

def zero_bold(bw_img):
    (horizontal, vertical) = bw_img.shape
    for horiz in bw_img:
        # if horiz[2] != horiz[3]:
        #     # print(horiz)
        #     print_this = True
        # else:
        #     print_this = False

        change_on_next = False
        for indx, val in enumerate(horiz):
            if change_on_next == True:
                # if last loop was change, set last pixel to black
                # print("before: ", horiz[indx-1])
                horiz[indx-1] = 0
                horiz[indx-2] = 0
                horiz[indx-3] = 0
                # print("after: ", horiz[indx-1])
                change_on_next = False
            elif indx > 2 and (val != horiz[indx-1]) and val == 255:
                change_on_next = True
                # print(str(vert_value))
                # check, but don't change till next pass
        # if print_this:
        #     print("after: ", horiz)

        # row_value_changes = 0
        # value_in_row = horiz[0]
        #     if vert_value != value_in_row:
        #         row_value_changes += 1
        #         value_in_row = vert_value
    return bw_img

# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\Isolated\\BlackandWhite\\Rotated")
def show_images(images, cols = 1, titles = None):
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

def custom_thresh_otsu(img):
    # block_size = int(.00001*(img.size))
    # while block_size % 2 != 1:
    #     block_size += 1
    # print("block size: {}".format(block_size))
    # blur = cv2.GaussianBlur(img,(5),0)
    blur = cv2.medianBlur(img,5)

    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    print("Imagethresh: ", thresh)
    return thresh

def get_region_quantile(outer_index, inner_index, full_image):
    '''TODO: finish creating this method for more dynamic functionality, but don't get caught in the weeds'''
    quantile_regions = 16 # must have desireable integer square root

    # this ^ devides the image into 16 squares
    image_height_quantile = len(full_image)/math.sqrt(quantile_regions)
    image_width_quantile = len(full_image[0])/math.sqrt(quantile_regions)


    height_quantile = round(outer_index / image_height_decile)
    width_quantile round(outer_index / image_height_decile)

    # number the quantiles left to right, top to bottom
    quantile = -1
    if height_quantile == 0:
        quantile = 1 + width_quantile
    if height_quantile == 1:
        quantile = 5 + width_quantile
    if height_quantile == 2:
        quantile = 9 + width_quantile
    if height_quantile == 3:
        quantile = 13 + width_quantile
    return quantile

def custom_gray_to_bw(img_gray, thresh_val, black_val=0, white_val=255, bold_value=0):
    '''
    goal here is to locate local threshold in the image, and convert to bw based on that locality.
    Ideal is probably 10 percent of image blocks
    '''
    # first, break image into 10 squares

    # identify threshold for each of 10 regions (squares)

    # convert to black and white depending on region threshold

    thresh_val += bold_value
    for ind1, val1 in enumerate(img_gray):
        for ind2, val2 in enumerate(img_gray[ind1]):
            if val2 > thresh_val + bold_value:
                img_gray[ind1][ind2] = white_val
            else:
                img_gray[ind1][ind2] = black_val
    return img_gray

def add_contours(img_source_bw, img_color):
    # Find the contours
    image,contours,hierarchy = cv2.findContours(img_source_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img_cont_unfiltered = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_cont_unfiltered,(x,y),(x+w,y+h),(0,255,0),5)
        if (22 <= w <= 500) and (22 <= h <= 200):
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
    return img_color, img_cont_unfiltered

for file in glob.glob("*.jpg"):
    img = cv2.imread(file)
    height, width, channels = img.shape

    unrefined_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert img to grayscale to prep for black and white
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    otsus_bi_thresh = custom_thresh_otsu(img_gray)

    # be more mathematical about this... if we can, we should convert to bw based on area of image.
    bold_value = .1*otsus_bi_thresh

    img_low_thresh = custom_gray_to_bw(img_gray, otsus_bi_thresh, 0, 255, bold_value)

    img_low_thresh,low_thresh_contours,low_thresh_hierarchy = cv2.findContours(img_low_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    img_low, img_cont_unfiltered = add_contours(img_low_thresh, unrefined_img)

    show_images([img_low_thresh, img_cont_unfiltered, img_low])


    # Convert to black and white based on automatice OTSU threshold
    # without black and white, area detection is very poor
    # adaptive threshhold could be used, but assuming images are evenly lit
    # (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, ((cv2.THRESH_BINARY | cv2.THRESH_OTSU) -25))

    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(img_gray,(1,1),0)
    # thresh, img_bw = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)



    # bold_value = 0
    # if 90 <= otsus_bi_thresh <= 130:
    # elif otsus_bi_thresh + .15*otsus_bi_thresh < 255:
    #     bold_value = .15*otsus_bi_thresh


    # Find the contours
    # image,contours,hierarchy = cv2.findContours(img_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it

    # img = add_contours(img_bw, img)
    # show_images([img_bw, img, img_low_thresh, unrefined_img])

# ignore every contour fully contained within another contour
# everything interesting is generally captured. Try combining overlapping contours when similar height
# TODO begin sorting through each box and eliminating non applicable contours
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
