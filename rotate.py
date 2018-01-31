import cv2
import os
import glob
import numpy as np
import imutils

os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\cleaner_selections")

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
        # print(vert_line_lengths[indx][1])
        if vert_line_lengths[indx][1] > best_angle_weight+1:
            best_angle = vert_line_lengths[indx][0]
            best_angle_weight = vert_line_lengths[indx][1]
            # print(vert_line_lengths[indx])
    return best_angle


for file in glob.glob("*.jpg"):
    print("reading img: ", file)
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = deskew(img)
    # print(long_contour_angle(img))
    img = imutils.rotate_bound(img, long_contour_angle(img))
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
