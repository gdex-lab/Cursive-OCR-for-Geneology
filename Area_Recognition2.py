import cv2
import pytesseract
import os
import glob
import cv2
import numpy as np

from PIL import Image


# print(pytesseract.pytesseract.tesseract_cmd)
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\grant\\AppData\\Local\\Tesseract-OCR'
# print(pytesseract.pytesseract.tesseract_cmd)
# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\Isolated")
os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\1900s")


# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\CensusRecords\\Isolated\\BlackandWhite\\Rotated")

def auto_bw_plus(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth the image to avoid noises
    # img_gray = cv2.medianBlur(img_gray,1)

    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(img_gray,255,1,1,11,2)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    # thresh = cv2.dilate(thresh,None,iterations = 15)
    # thresh = cv2.erode(thresh,None,iterations = 15)
    print("Image threshold = ", thresh)
    return thresh_color #img_bw

for file in glob.glob("*.jpg"):
    # img = cv2.imread(file)
    # img = Image.open(file) #pytesseract
    img = cv2.imread(file) #opencv

    cv2.imshow('img', auto_bw_plus(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow('img',remove_gray(img, 255-thresh, 255+thresh))
    # print(img)
    # text = pytesseract.image_to_string(img, lang = 'eng')
    # print(text.encode('utf-8'))




    # Bold experiment
    # show_images([img_bw, zero_bold(img_bw)])
    # plt.imshow(img)
    # plt.show()
    # time.sleep(10)


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
