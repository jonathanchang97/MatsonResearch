import numpy as np
import cv2
import math

img = cv2.imread("frames(20633-20650)/scene00001.png", 0)
height, width = img.shape[0:2]

thresh = 75
# change to easier to deal with black and white image
ret, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

# eliminate noise in image
kernel = np.ones((7,7), 'uint8')
# remove noise in image and leave only the circles, erodes, then dilates
imgOpening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# maybe dilate a bit since the fitted ellipse is smaller
#imgOpening = cv2.dilate(thresh, np.ones((3,3), 'uint8'), iterations=1)


# contours on adjusted image to create shape
contours, hierarchy = cv2.findContours(imgOpening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imgContours = img.copy()
index = -1
thickness = 3
color = (255, 0, 255)


imgBlack = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)

for c in contours:
    ellipse = cv2.fitEllipse(c)
    ellipseImage = cv2.ellipse(img,ellipse,color,2)
    ellipseBlackImage = cv2.ellipse(imgBlack,ellipse,color,2)
    # center, axis_length and orientation of ellipse
    (center,axes,orientation) = ellipse
    # length of MAJOR and minor axis
    majoraxis_length = max(axes)
    minoraxis_length = min(axes)
    # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
    eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
    # area of ellipse
    area = math.pi*(majoraxis_length/2)*(minoraxis_length/2)
    print("Area: {}, eccentricity, {}".format(area, eccentricity))
    hull = cv2.convexHull(c)
    # defects = cv2.convexityDefects(c,hull)
    area = cv2.contourArea(hull)
    print("Convex Area: {}".format(area))
    #convexImage = c.convexImage(c)
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(ellipseImage,[box],0,color,2)
    # for i in range(defects.shape[0]):
    #     s,e,f,d = defects[i,0]
    #     start = tuple(c[s][0])
    #     end = tuple(c[e][0])
    #     far = tuple(c[f][0])
    #     cv2.line(ellipseBlackImage,start,end,color,2)
    #     # for comparison purposes
    #     cv2.line(ellipseImage,start,end,color,2)
    #     #cv2.circle(imgBlack,far,5,color,-1)



objects = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)
# contours on fitted ellipse
contours2, hierarchy2 = cv2.findContours(imgOpening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours2:
    cv2.drawContours(objects, [c], -1, color, -1)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(objects,(cx,cy), 4, (0,0,255), -1)

    print("Area: {}, perimeter, {}".format(area, perimeter))



cv2.imwrite("segmented-image.png", imgOpening)
cv2.imwrite("ellipse-image.png", ellipseImage)
cv2.imwrite("ellipse-black-image.png", ellipseBlackImage)
#cv2.imwrite("convex-black-image.png", convexImage)
# cv2.imwrite("contoured-image.png", objects)

