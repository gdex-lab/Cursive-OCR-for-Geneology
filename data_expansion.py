import cv2
import os, glob


path="C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset\\X - enlarged, rotated"
os.chdir(path)
for file in glob.glob("*.jpg"):
    image = cv2.imread(file)
    larger = cv2.resize(image, (80, 69))
    rows,cols = larger.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),7,1)
    larger_rotated = cv2.warpAffine(larger,M,(cols,rows))
    larger_rotated_cropped = larger_rotated[4:64, 5:75]
    # print(larger.shape)
    # print(larger_rotated.shape)
    # print(larger_rotated_cropped.shape)
     # = warpAffine(larger, dst, r, Size(src.cols, src.rows))
    # cv2.imshow('output', larger_rotated_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(file, larger_rotated_cropped)
