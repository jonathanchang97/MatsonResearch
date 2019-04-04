import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt

def getListOfFiles(dirName):
    listOfFile = os.listdir('frames(20633-20650)')
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

def processImage(img):
    height, width = img.shape[0:2]

    thresh = 68
    # change to easier to deal with black and white image
    ret, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # eliminate noise in image
    kernel = np.ones((7,7), 'uint8')
    # remove noise in image and leave only the circles, erodes, then dilates
    imgOpening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


    # contours on adjusted image to create shape
    contours, hierarchy = cv2.findContours(imgOpening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    imgContours = img.copy()
    index = -1
    thickness = 3
    color = (255, 0, 255)


    #objects = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)
    imgBlack = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)


    for c in contours:
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        print("Convex Area: {}".format(area))

        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(c[s][0])
                end = tuple(c[e][0])
                far = tuple(c[f][0])
                cv2.line(imgBlack,start,end,color,2)
                # for comparison purposes
                cv2.line(img,start,end,color,2)
                #cv2.circle(imgBlack,far,5,color,-1)



    cv2.imwrite("segmented-image.png", imgOpening)
    cv2.imwrite("contour-on-original-image.png", img)
    cv2.imwrite("contour-on-black-image.png", imgBlack)
    # cv2.imwrite("fixed-contour-image.png", imgContours)

    return area

def main():
    dirName = 'frames(20633-20650)';
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)

    # for files in listOfFiles:
    #     img = cv2.imread(files, 0)
    #     area = processImage(img)

    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show()
    plt.savefig('plots/basic-plot.png')
    

if __name__ == '__main__':
    main()






# imgBlack = cv2.cvtColor(imgBlack, cv2.COLOR_BGR2GRAY)
# contours2, hierarchy2 = cv2.findContours(imgBlack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# objects2 = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)

# for c in contours2:
#     cv2.drawContours(imgContours, [c], -1, color, -1)

#     # area, perimeter, etc
#     area = cv2.contourArea(c)
#     perimeter = cv2.arcLength(c, True)

#     M = cv2.moments(c)
#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])
#     cv2.circle(imgContours,(cx,cy), 4, (0,0,255), -1)

#     print("Area: {}, perimeter, {}".format(area, perimeter))