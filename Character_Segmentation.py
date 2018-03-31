"""RATHER THAN...
segmenting out so you can convolve, just jump to the convolutional model now!
You can add the rotated and resized individual letters to your data set for variation ( do this step after building the dataset ).
May need some high-level segmentation, like record type for increased accuracy, but just get as accurate as you can at a generic level first.
Save the rawest of the raw crops (with ligatures reaching edge) for model training.
"""



# Could use text area detection for average letter size
# TODO experiment with accuracy from searching entire image for each letter
# TODO for additional accuracy, ask user to select one of each common letter?


# templates work, but must be the same size as the letters in the image
# also, many templates would be necessary for each letter. A comparison could be made for highest thresh in that section of image
 # TODO COMBINE TECHNIQUES FOR HIGHEST ACCURACY?

# TODO could use the convolving template to identify characters, and then crop those to be passed through convolutional network in original form!
# could use ten templates of each character to segment and devide

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
import imutils
# os.chdir("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\rotated_crops\\clean_selections")
os.chdir("C:\\Users\\grant\\IS\\IS552\\JSPapersBookofTheLawoftheLord\\RotationsApplied")
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


def get_location_list_position(locations_list, point):
    position = len(locations_list)
    for indx, val in enumerate(locations_list):
        position = indx
        # If new coordinates are more than 5 pixels diffent in any position, than add position
        if abs(point[0] - val[0]) > 4 or abs(point[1] - val[1]) > 4:
            position += 1
        else:
            break

    return position

for file in glob.glob("*.jpg"):
    '''based on cropped image height, set 3 letter sizes, around 1/3 the cropped image Height
        convolve over the image comparing templates (one of each size) with letters
        Establish votes for by each letter

        original img remains 3 channels the entire time.


    '''
    print("reading img: ", file)
    img = cv2.imread(file)

    img_height, img_width = img.shape[:2]
    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_count = 0
    sizes = [.98, 1, 1.02]
    rotations = [0,358, 2]
    template_imgs = []
    a_votes = [0]
    a_locations = []
    # for template_img in glob.glob("C:\\Users\\grant\\IS\\Past\\IS693R\\image_project\\images\\misc\\rotated_crops\\hhlettersrotated\\nadaNADA*.jpg"):
    for template_img in glob.glob("C:\\Users\\grant\\IS\\IS552\\JSPapersBookofTheLawoftheLord\\templates\\*.jpg"):
        img_count += 1
        print(img_count)
        template = cv2.imread(template_img,0)
        # template = cv2.adaptiveThreshold(template,255,cv2.cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
        # ret,template = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
        # template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        # img_low_thresh3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,window_block_neighbors,2)
        template_height, template_width = template.shape[:2]
        # print("Template shape: ", template.shape)
        # print("Img gray shape: ", img_gray.shape)
        for angle in rotations:
            template_rotated = imutils.rotate(template, angle=angle)
            for resize in sizes:
                # print("new width: ",int((resize*img_height)/(template_height/template_width)))
                # print("New height: ", int(resize*img_height))
                template_rotated = cv2.resize(template_rotated, (int(resize*template_width), int(resize*template_height)), interpolation = cv2.INTER_CUBIC)
                resized_template_w, resized_template_h = template_rotated.shape[:2]
                res = cv2.matchTemplate(img,template_rotated,cv2.TM_CCOEFF_NORMED)
                # print(img.shape)
                threshold = 0.8
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    list_location_postition = get_location_list_position(a_locations, pt)
                    # print(pt)
                    # print(l ist_location_postition)
                    if len(a_votes) <= list_location_postition:
                        a_votes.append(1)
                    else:
                        a_votes[list_location_postition] +=1
                    if len(a_locations) <= list_location_postition:
                        a_locations.append(pt)
                    cv2.rectangle(img, pt, (pt[0] + resized_template_w, pt[1] + resized_template_h), (0,0,255-10*img_count), 2)
                # print(img.shape)
            # if img_count %10 == 1:
            # template_imgs.append(template)
                # cv2.imshow('img', template_rotated)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    cv2.putText(img, str(a_votes), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)
    template_imgs.append(img)
    show_images(template_imgs, 1)
