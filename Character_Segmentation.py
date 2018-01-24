# TODO (optional?) slim all crops down to the single pixel level. <- this may help for uniform join identification
# TODO identify joints.
# TODO make divisions based on separation between joint clusters.

# a few hard fast rules to help:
    # any space with two separate line in the same vertical, is a letter, not a ligature.
    # all outgoing (besides B's) connectors are from the top or bottom of the letter
    # round or straight, every character has walls
    # eroding horizontal lines will leave character walls



# Make the best splits you can, but give multiple options. Then train your algorithm to see which words make the most sense based on splits.

import cv2
import os
import glob
import numpy as np

os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\Words_Isolated")

def cv_imshow(img):
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for file in glob.glob("*.png"):
    print("reading img: ", file)
    img = cv2.imread(file)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/90,100,minLineLength,maxLineGap)
    print(lines[0])
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    # cv2.imwrite('houghlines5.jpg',img)
    cv_imshow(img)
    # difference between input image and Opening of the image
    # kernel = np.ones((3,3),np.uint8)
    # # blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    # #tophat
    # highlights = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print("thresh")
    # cv_imshow(thresh)
    #
    # kernel = np.ones((1,2),np.uint8)
    # erosion = cv2.erode(thresh,kernel,iterations = 6)
    # print("erosion")
    # cv_imshow(erosion)

    # noise removal
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # print("opening")
    # cv_imshow(opening)
    # # sure background area
    # sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # print("sure_bg")
    # cv_imshow(sure_bg)
    #
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # print("sure_fg")
    # cv_imshow(sure_fg)
    #
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)
    # print("unknown")
    # cv_imshow(unknown)
    #
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    #
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    #
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0
    #
    # markers = cv2.watershed(img,markers)
    # img[markers == -1] = [255,0,0]
    # print("img")
    # cv_imshow(img)
    #
    # print("markers")
    # cv_imshow(markers)
