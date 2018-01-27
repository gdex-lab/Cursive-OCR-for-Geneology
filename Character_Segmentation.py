# TODO (optional?) slim all crops down to the single pixel level. <- this may help for uniform join identification
# TODO identify joints.
# TODO make divisions based on separation between joint clusters.

# Could use text area detection for average letter size
# TODO experiment with accuracy from searching entire image for each letter
# TODO for additional accuracy, ask user to select one of each common letter?



# templates work, but must be the same size as the letters in the image
# also, many templates would be necessary for each letter. A comparison could be made for highest thresh in that section of image
 # TODO COMBINE TECHNIQUES FOR HIGHEST ACCURACY?


# a few hard fast rules to help:
    # any space with two separate line in the same vertical, is a letter, not a ligature.
    # all outgoing (besides B's) connectors are from the top or bottom of the letter
    # round or straight, every character has walls
    # eroding horizontal lines will leave character walls



# Make the best splits you can, but give multiple options. Then train your algorithm to see which words make the most sense based on splits.
import matplotlib.pyplot as plt
import cv2
import os
import glob
import numpy as np

os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\cleaner_selections")

def cv_imshow(img):
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

def find_vert_parallels(img):
    print(img)
    height, width, channels = img.shape
    print("Height: {}".format(height))
    print("Width: {}".format(width))
    for col in img:
        row = 0
        while row < height:
            print(col[row])
            row+=1
def get_location_list_position(locations_list, point):
    position = len(locations_list)
    for indx, val in enumerate(locations_list):
        position = indx
        # If new coordinates are more than 5 pixels diffent in any position, than add position
        print(point, val)
        if abs(point[0] - val[0]) > 4 or abs(point[1] - val[1]) > 4:
            position += 1
        else:
            break

    return position

for file in glob.glob("*.jpg"):
    '''based on cropped image height, set 3 letter sizes, around 1/3 the cropped image Height
        convolve over the image comparing templates (one of each size) with letters
        Establish votes for by each letter

    '''
    print("reading img: ", file)
    img = cv2.imread(file)
    height, width, channels = img.shape
    # find_vert_parallels(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_count = 0
    sizes = [.25, .35, .45, .55, .65 ,.75]
    template_imgs = []
    a_votes = [0]
    a_locations = []
    for template_img in glob.glob("..\\individual_letters\\a\\*.jpg"):
        img_count += 1
        print(img_count)
        template = cv2.imread(template_img,0)
        template_height, template_width = template.shape

        for resize in sizes:
            # Keep in mind that the resized images are greyscaled, which will require additional felexibility in threshold
            template = cv2.resize(template, (int((resize*height)/(template_height/template_width)), int(resize*height)))

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.75
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                list_location_postition = get_location_list_position(a_locations, pt)
                print(pt)
                print(list_location_postition)
                if len(a_votes) <= list_location_postition:
                    a_votes.append(1)
                else:
                    a_votes[list_location_postition] +=1
                if len(a_locations) <= list_location_postition:
                    a_locations.append(pt)
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255-10*img_count), 2)
    # img = cv2.resize(img, fx=2, fy=2)
    cv2.putText(img, str(a_votes), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)
    template_imgs.append(img)
    show_images(template_imgs)
    # break
    # edges = cv2.Canny(gray,50,150,apertureSize = 3)
    # minLineLength = 5
    # maxLineGap = 5
    # lines = cv2.HoughLinesP(edges,1,np.pi/90,100,minLineLength,maxLineGap)
    # print(lines[0])
    # for x1,y1,x2,y2 in lines[0]:
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    #
    # # cv2.imwrite('houghlines5.jpg',img)
    # cv_imshow(img)
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
    # kernel = np.ones((6,1),np.uint8)
    # erosion = cv2.erode(thresh,kernel,iterations = 1)
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
