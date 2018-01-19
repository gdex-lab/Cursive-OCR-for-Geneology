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

def get_region_quantile(outer_index, inner_index, height, width, quantile_regions=16):
    '''
    params:
        outer_index: the outer index of pixel location, e.g. 0 from image[0][1]
        inner_index: the inner index of pixel location, e.g. 1 from image[0][1]
        height: image height, as calculated by image.shape
        width: image width, as calculated by image.shape
        quantile_regions: number of desired quantiles, must have whole integer sqrt

    Takes quantile regions number and calculates the quantile in which each pixel lands.
    Numbering is from left to right, top to bottom.
    '''
    q_sqrt = math.sqrt(quantile_regions)

    if q_sqrt % 1 != 0:
        print('Your quantile regions do not have a whole number as a square root')
        sys.exit(1)

    image_height_quantile = height/q_sqrt
    image_width_quantile = width/q_sqrt

    height_quantile = round(outer_index / image_height_decile)
    width_quantile = round(outer_index / image_height_decile)

    quantile = height_quantile * q_sqrt + width_quantile + 1
    return quantile

def custom_gray_to_bw(img_gray, thresh_val, black_val=0, white_val=255, bold_value=0):
    '''
    goal here is to locate local threshold in the image, and convert to bw based on that locality.
    Ideal is probably 10 percent of image blocks
    '''
    # first, break image into 16 squares

    # identify threshold for each of 16 regions (squares)
    quantile_regions = 16
    height, width = img_gray.shape
    q_sqrt = int(math.sqrt(quantile_regions))
    image_height_quantile = height/q_sqrt
    image_width_quantile = width/q_sqrt

    quantiles_list = []
    # q1 = img_gray[0:image_height_quantile, 0:image_width_quantile]
    for i in range(q_sqrt):
        for ii in range(q_sqrt):
            quantiles_list.append([[i*image_height_quantile, image_height_quantile],[ii*image_width_quantile, image_width_quantile]])
    print(height, width, image_height_quantile, image_width_quantile, quantiles_list)
    # convert to black and white depending on region threshold

    thresholds_list = []
    for quantile in quantiles_list:
        thresholds_list.append(cv2.adaptiveThreshold(img_gray[quantile[0], quantile[1]],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2))
    # calculate local thresholding

    thresh_val += bold_value
    for ind1, val1 in enumerate(img_gray):
        for ind2, val2 in enumerate(img_gray[ind1]):
            if val2 > thresh_val + bold_value:
                img_gray[ind1][ind2] = white_val
            else:
                img_gray[ind1][ind2] = black_val
    return img_gray


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
