import cv2
import pytesseract
import os
import glob
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

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
    plt.show()

for file in glob.glob("*.jpg"):
    img = cv2.imread(file)
    height, width, channels = img.shape

    # Convert img to grayscale to prep for black and white
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to black and white based on automatice OTSU threshold
    # without black and white, area detection is very poor
    # adaptive threshhold could be used, but assuming images are evenly lit
    # (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, ((cv2.THRESH_BINARY | cv2.THRESH_OTSU) -25))

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_gray,(1,1),0)
    thresh, img_bw = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    show_images([img_bw, zero_bold(img_bw)])
    plt.imshow(img)
    plt.show()
    time.sleep(10)
    # image block size 21 worked, so did 41
    # block_size = 41
    # Apply adaptive threshold
    # thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    # thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,block_size,0)

    # smooth the image to avoid noises
    # thresh = cv2.medianBlur(img_bw,1)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    # img_bw = cv2.dilate(thresh,None,iterations = 5)
    # img_bw = cv2.erode(thresh,None,iterations = 5)

#     # Find the contours
#     image,contours,hierarchy = cv2.findContours(img_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#     unrefined_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # For each contour, find the bounding rectangle and draw it
#     for cnt in contours:
#         x,y,w,h = cv2.boundingRect(cnt)
#         cv2.rectangle(unrefined_img,(x,y),(x+w,y+h),(150,205,0),5)
#         # print(x,y,w,h)
#
#
#         if (22 <= w <= 500) and (22 <= h <= 200):
#             # """iterate through contours and check for variety in the images"""
#             crop_img = img_bw[y:y+h, x:x+w]
#             # horizontal = columns, vertical = rows
#             (horizontal, vertical) = crop_img.shape
#             h_lower = int(.2 * horizontal)
#             h_upper = int(.8 * horizontal)
#             v_lower = int(.2 * vertical)
#             v_upper = int(.8 * vertical)
#
#             # sum total changes accross the bw array. If none of middle sixty percentof  arrays contain more than one color change, discard entire boundary
#             contained_variety = 0
#             for horiz in crop_img[h_lower:h_upper]:
#                 row_value_changes = 0
#                 value_in_row = horiz[0]
#                 for vert_value in horiz[v_lower:v_upper]:
#                     if vert_value != value_in_row:
#                         row_value_changes += 1
#                         value_in_row = vert_value
#                 if row_value_changes > 2: # a change and back (we want more than one line)
#                     contained_variety += 1
#             if contained_variety > 0:
#                 # cv2.imshow("cropped", crop_img)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#                 # time.sleep(1)
#                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
#
#
#     show_images([img_bw, unrefined_img, img])
#     # plt.imshow(img)
#     # plt.show()
#     # plt.imshow(unrefined_img)
#     # plt.show()
#     # plt.show()
#     # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('img', 1200,800)
#     # cv2.imshow('img',img_bw)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#
# # TODO begin sorting through each box and eliminating non applicable contours
# # next, need to refine boxes to include whole words, where possible.
# # maybe combine horizontally similar boxes
# # maybe move forward with current results to better understand next steps before spending too much time refining
