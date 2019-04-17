import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
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
        1.0, color, 1)
 
    # return the image with the contour number drawn on it
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
    # brightness = 20
    # contrast = 50
    # imgContrast = apply_brightness_contrast(img, brightness, contrast)
    imgContrast = img


    thresh = 65
    # change to easier to deal with black and white image
    ret, thresh = cv2.threshold(imgContrast, thresh, 255, cv2.THRESH_BINARY)

    # eliminate noise in image
    kernel = np.ones((7,7), 'uint8')
    # fills in holes in drop shape image that are from the obstructions
    imgClosed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    imgClosed = cv2.morphologyEx(imgClosed, cv2.MORPH_CLOSE, kernel)
    # remove noise in image and leave only the circles, erodes, then dilates
    imgOpening = cv2.morphologyEx(imgClosed, cv2.MORPH_OPEN, kernel)

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
        ellipse = cv2.fitEllipse(c)
        ellipseImage = cv2.ellipse(img,ellipse,color,1)
        ellipseBlackImage = cv2.ellipse(imgBlack,ellipse,color,1)
        # center, axis_length and orientation of ellipse
        (center,axes,orientation) = ellipse
        # length of MAJOR and minor axis
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
        eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
        # area of ellipse
        area = math.pi*(majoraxis_length/2)*(minoraxis_length/2)

        areaList.append(area)
        # print("Ellipse Area: {}".format(area))

        img = draw_num_on_contour(img, c, i, color)
        imgBlack = draw_num_on_contour(imgBlack, c, i, color)

        i += 1;

    cv2.imwrite("processed-images/contrast-image" + num + ".png", imgContrast)
    cv2.imwrite("processed-images/segmented-image" + num + ".png", imgOpening)
    cv2.imwrite("processed-images/contour-on-original-image" + num + ".png", img)
    # cv2.imwrite("processed-images/contour-on-black-image" + num + ".png", imgBlack)
    # cv2.imwrite("fixed-contour-image.png", imgContours)

    return areaList


def main():
    dirName = 'frames/reducedFO-16_cyc11';
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)

    # img = cv2.imread("frames20633-20650/scene00001.png", 0)
    # areaList = processImage(img, "1")
    # for a in areaList:
    #     print("Convex Area: {}".format(a))
    i = 0;
    areaList = []

    for file in listOfFiles:
        img = cv2.imread(file, 0)
        if img is None:
            print("Could not open or find the image " + file + "!\n")
            continue
        
        areas = processImage(img, str(i + 1))
        areaList.append(areas)
        print("Image: {}".format(file))
        for a in areaList[i]:
            print("Convex Area: {}".format(a))
        i += 1;

    timeList1 = []
    timeList2 = []
    numImages = len(listOfFiles)
    startTime = 52.5
    endTime = 57.5
    time = 52.5
    timestep = (endTime - startTime) / numImages

    for x in range(numImages):
        timeList1.append(time)
        timeList2.append(time + timestep)
        time += timestep
        if len(areaList[x]) < 3 or areaList[x][0] <= 1800:
            areaList[x][0] = (areaList[x-1][0])

    areaList1 = [item[0] for item in areaList]
    areaList2 = [item[2] for item in areaList]
            

    plt.plot(timeList1, areaList1, label = "top")
    plt.plot(timeList2, areaList2, label = "bottom")
    plt.xlabel('time')
    plt.ylabel('area (pixels)')
    plt.title('top-view-' + dirName[7:])
    plt.savefig('plots/top-view-' + dirName[7:] + '.png')

    # plt.plot([1,2,3,4])
    # plt.ylabel('some numbers')
    # plt.show()
    # plt.savefig('plots/basic-plot.png')
    

if __name__ == '__main__':
    main()
