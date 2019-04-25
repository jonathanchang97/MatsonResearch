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


def draw_defect_lines_on_contour(img, c, defects, color):
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(c[s][0])
            end = tuple(c[e][0])
            far = tuple(c[f][0])
            cv2.line(img,start,end,color,1)
            #cv2.circle(imgBlack,far,5,color,-1)
    return img


def processImage(img, num):
    #crop image to top view only
    # y = 115
    # x = 30
    # h = 250
    # w = 300
    # img = img[y:y+h, x:x+w]

    color = (255, 0, 255)
    imgContours = img.copy()

    # Increase contrast in the image
    # brightness = 20
    # contrast = 50
    # imgContrast = apply_brightness_contrast(img, brightness, contrast)
    #imgContrast = img

    # Gaussian blur to reduce noise and accuracy
    imgBlur = cv2.GaussianBlur(img,(5,5),0)

    thresh = 55
    # change to easier to deal with black and white image
    ret, thresh = cv2.threshold(imgBlur, thresh, 255, cv2.THRESH_BINARY)

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


    imgBlack = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)

    areaList = []
    i = 0

    # Find convexity defects and store area along with the images
    for c in cnts:
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)
        hull = cv2.convexHull(c, False)
        area = cv2.contourArea(hull)

        areaList.append(area)
        # print("Convex Area: {}".format(area))

        img = draw_defect_lines_on_contour(img, c, defects, color)
        cv2.drawContours(imgBlack, [hull], -1, color, -1)
        #imgBlack = draw_defect_lines_on_contour(imgBlack, c, defects, color)

        img = draw_num_on_contour(img, c, i, color)
        imgBlack = draw_num_on_contour(imgBlack, c, i, color)

        i += 1;


    cv2.imwrite("processed-images/contour-on-black-image" + num + ".png", imgBlack)
    imgBlack = cv2.imread("processed-images/contour-on-black-image" + num + ".png", 0)


    # # For finding the circle
    # # Find the contours of the new corrected image
    # cnts, hierarchy = cv2.findContours(imgBlack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Get rid of any lingering small contours and only keep the 3 of interest
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    # # sort the contours according to the provided method
    # (cnts, boundingBoxes) = sort_contours(cnts)

    # widthList = []

    # # Find convexity defects and store area along with the images
    # for c in cnts:
    #     (x,y),radius = cv2.minEnclosingCircle(c)
    #     center = (int(x),int(y))
    #     radius = radius
    #     cv2.circle(imgBlack,center,int(radius),color,1)

    #     # x,y,w,h = cv2.boundingRect(c)
    #     widthList.append(radius)
    #     # cv2.rectangle(imgBlack,(x,y),(x+w,y+h),color,1)

    # #cv2.imwrite("processed-images/contrast-image" + num + ".png", img)
    # #cv2.imwrite("processed-images/segmented-image" + num + ".png", imgOpening)
    # cv2.imwrite("processed-images/contour-on-original-image" + num + ".png", img)
    # cv2.imwrite("processed-images/contour-on-black-image" + num + ".png", imgBlack)
    # # cv2.imwrite("fixed-contour-image.png", imgContours)

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
        if len(areaList[x]) < 3:
            areaList[x][0] = (areaList[x-1][0])

    areaList1 = [item[0] for item in areaList]
    areaList2 = [item[2] for item in areaList]

    plotList1 = list(zip(timeList1, areaList1))
    plotList2 = list(zip(timeList2, areaList2))

    # plotListAdjusted1 = [item for item in plotList1 if item[1] > 2375]
    # plotListAdjusted2 = [item for item in plotList2 if item[1] > 2375]

    plotListAdjusted1 = [plotList1[0]]
    plotListAdjusted2 = [plotList2[0]]

    pointsLength = len(plotList1) - 1
    for x in range(pointsLength):
        if abs(plotList1[x][1] - plotList1[x+1][1]) < 60:
            plotListAdjusted1.append(plotList1[x+1])
        if abs(plotList2[x][1] - plotList2[x+1][1]) < 60:
            plotListAdjusted2.append(plotList2[x+1])

    x1 = [] 
    y1 = []
    x2 = []
    y2 = []
    x1 = [item[0] for item in plotList1]
    y1 = [item[1] for item in plotList1]
    x2 = [item[0] for item in plotList2]
    y2 = [item[1] for item in plotList2]


    # plt.plot(timeList1, areaList1, label = "top")
    # plt.plot(timeList2, areaList2, label = "bottom")
    plt.plot(x1, y1, label = "top")
    plt.plot(x2, y2, label = "bottom")
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
