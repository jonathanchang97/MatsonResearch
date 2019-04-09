import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt


def getListOfFiles(dirName):
    listOfFile = os.listdir('frames20633-20650')
    allFiles = list()
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles   


def apply_brightness_contrast(input_img, brightness, contrast):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def sort_contours(cnts):
    # initialize the reverse flag and sort index
    reverse = False
    i = 1
 
    # # handle if we need to sort in reverse
    # if method == "right-to-left" or method == "bottom-to-top":
    #     reverse = True
 
    # # handle if we are sorting against the y-coordinate rather than
    # # the x-coordinate of the bounding box
    # if method == "top-to-bottom" or method == "bottom-to-top":
    #     i = 1
 
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i]+b[1][3], reverse=reverse))
 
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def draw_num_on_contour(img, c, i, color):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
    # draw the countour number on the image
    cv2.putText(img, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, color, 2)
 
    # return the image with the contour number drawn on it
    return img


def draw_defect_lines_on_contour(img, c, defects, color):
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(c[s][0])
            end = tuple(c[e][0])
            far = tuple(c[f][0])
            cv2.line(img,start,end,color,2)
            #cv2.circle(imgBlack,far,5,color,-1)
    return img


def processImage(img, num):
    # crop image to top view only
    # y = 115
    # x = 30
    # h = 250
    # w = 300
    # img = img[y:y+h, x:x+w]

    thickness = 3
    color = (255, 0, 255)
    imgContours = img.copy()

    # Increase contrast in the image
    brightness = 50
    contrast = 50
    imgContrast = apply_brightness_contrast(img, brightness, contrast)
    # contrast = 40

    # f = 131*(contrast + 127)/(127*(131-contrast))
    # alpha_c = f
    # gamma_c = 127*(1-f)

    # imgContrast = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)


    thresh = 65
    # change to easier to deal with black and white image
    ret, thresh = cv2.threshold(imgContrast, thresh, 255, cv2.THRESH_BINARY)

    # eliminate noise in image
    kernel = np.ones((7,7), 'uint8')
    # remove noise in image and leave only the circles, erodes, then dilates
    imgOpening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # contours on adjusted image to create shape
    cnts, hierarchy = cv2.findContours(imgOpening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get rid of any lingering small contours and only keep the 3 of interest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

    # sort the contours according to the provided method
    (cnts, boundingBoxes) = sort_contours(cnts)


    #objects = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)
    imgBlack = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)

    areaList = []
    i = 0

    for c in cnts:
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)


        areaList.append(area)
        # print("Convex Area: {}".format(area))

        img = draw_defect_lines_on_contour(img, c, defects, color)
        imgBlack = draw_defect_lines_on_contour(imgBlack, c, defects, color)

        img = draw_num_on_contour(img, c, i, color)
        imgBlack = draw_num_on_contour(imgBlack, c, i, color)

        i += 1;

    cv2.imwrite("processed-images/contrast-image" + num + ".png", imgContrast)
    cv2.imwrite("processed-images/segmented-image" + num + ".png", imgOpening)
    cv2.imwrite("processed-images/contour-on-original-image" + num + ".png", img)
    cv2.imwrite("processed-images/contour-on-black-image" + num + ".png", imgBlack)
    # cv2.imwrite("fixed-contour-image.png", imgContours)

    return areaList


def main():
    dirName = 'frames20633-20650';
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)

    # img = cv2.imread("frames20633-20650/scene00001.png", 0)
    # areaList = processImage(img, "1")
    # for a in areaList:
    #     print("Convex Area: {}".format(a))

    i = 1;

    for file in listOfFiles:
        img = cv2.imread(file, 0)
        if img is None:
            print("Could not open or find the image " + file + "!\n")
            continue
    
        areaList = processImage(img, str(i))
        print("Image: {}".format(file))
        for a in areaList:
            print("Convex Area: {}".format(a))
        i += 1;


    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show()
    plt.savefig('plots/basic-plot.png')
    

if __name__ == '__main__':
    main()
