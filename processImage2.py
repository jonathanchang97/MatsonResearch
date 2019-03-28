import numpy as np
import cv2

import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

img = cv2.imread("frames(20633-20650)/scene00001.png", 0)
height, width = img.shape[0:2]

thresh = 75

#adaptiveThresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 215, 1)

# change to easier to deal with black and white image
ret, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)


# Take a look at Opening and Closing functions istead of erode

# eliminate noise in image
kernel = np.ones((7,7), 'uint8')
# erode away bad parts of the image
#imgEroded = cv2.erode(thresh, kernel, iterations=1)
# dilate the circles back to size before erosion
#imgDilated = cv2.dilate(imgEroded, kernel, iterations=1)
# remove noise in image and leave only the circles
imgOpening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# Look at Canny edge detection
#edges = cv2.Canny(img, 10, 80)
#cv2.imwrite("cannyimage.png", edges)

cimg = cv2.cvtColor(imgOpening,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(imgOpening,cv2.HOUGH_GRADIENT,1,30,
                            param1=50,param2=15,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imwrite("edgesimage.png", cimg)

# # Load picture and detect edges
# edges = canny(imgOpening, sigma=3, low_threshold=10, high_threshold=50)


# # Detect two radii
# hough_radii = np.arange(20, 35, 2)
# hough_res = hough_circle(edges, hough_radii)

# # Select the most prominent 5 circles
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                            total_num_peaks=2)

# # Draw them
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
# imgCircles = color.gray2rgb(imgOpening)
# for center_y, center_x, radius in zip(cy, cx, radii):
#     circy, circx = circle_perimeter(center_y, center_x, radius)
#     imgCircles[circy, circx] = (220, 20, 20)

# cv2.imwrite("edgesimage.png", imgCircles)



#detect edges

edges = canny(imgOpening, sigma=2.0,
              low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
img[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)


cv2.imwrite("hough-original-image.png", img)
cv2.imwrite("hough-black-image.png", edges)




# # contours on adjusted image to create shape
# contours, hierarchy = cv2.findContours(imgOpening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# imgContours = img.copy()
# index = -1
# thickness = 3
# color = (255, 0, 255)









# contours2, hierarchy2 = cv2.findContours(imgBlack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# objects2 = np.zeros([imgOpening.shape[0], imgOpening.shape[1], 3], np.uint8)

# for c in contours2:
#     cv2.drawContours(objects2, [c], -1, color, -1)

#     # area, perimeter, etc
#     area = cv2.contourArea(c)
#     perimeter = cv2.arcLength(c, True)

#     M = cv2.moments(c)
#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])
#     cv2.circle(objects2,(cx,cy), 4, (0,0,255), -1)

#     print("Area: {}, perimeter, {}".format(area, perimeter))


# cv2.imwrite("segmented-image.png", imgOpening)
# cv2.imwrite("contour-image.png", objects)
# cv2.imwrite("contour-on-original-image.png", img)
# cv2.imwrite("contour-on-black-image.png", imgBlack)
# cv2.imwrite("fixed-contour-image.png", objects2)

