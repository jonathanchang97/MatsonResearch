import numpy as np
import cv2


# img = cv2.imread("frames(20633-20650)/scene00001.png", 0)
img = cv2.imread("scene00001.png", 0)
height, width = img.shape[0:2]

thresh = 75

#adaptiveThresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 215, 1)
ret, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)


# erode away bad parts of the image
kernel = np.ones((7,7), 'uint8')
imgEroded = cv2.erode(thresh, kernel, iterations=1)
# Take a look at Opening and Closing functions istead of erode

# contours
# Look at Canny edge detection
contours, hierarchy = cv2.findContours(imgEroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imgContours = img.copy()
index = -1
thickness = 3
color = (255, 0, 255)

# area, perimeter, etc
objects = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')
for c in contours:
	cv2.drawContours(objects, [c], -1, color, -1)

	area = cv2.contourArea(c)
	perimeter = cv2.arcLength(c, True)

	M = cv2.moments(c)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	cv2.circle(objects,(cx,cy), 4, (0,0,255), -1)

	print("Area: {}, perimeter, {}".format(area, perimeter))

#cv2.drawContours(imgContours, contours, index, color, thickness)


cv2.imwrite("segmentedimage.png", imgEroded)
cv2.imwrite("contourimage.png", objects)





# img.shape
# img.dtype

#ty = type(img)

# le = len(img)


#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

#cv2.imshow("Image",img)

#cv2.waitKey(0)

