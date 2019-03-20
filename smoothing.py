#!/usr/bin/python

# Standard imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read image
im = cv2.imread("/home/cesar/Desktop/blob.jpg", cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 256
#params.minThreshold = 10
#params.maxThreshold = 200
# Filter by Area.
params.filterByArea = True
#params.minArea = 1500
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
#params.minConvexity = 0.87
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

im_inverse = 255-im
# Create a detector with the parameters

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255, 0, 0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs

ret, thresh = cv2.threshold(im_inverse,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# You need to choose 4 or 8 for connectivity type
connectivity = 4
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
print(num_labels)

# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
print()
print(stats)
# The fourth cell is the centroid matrix
centroids = output[3]
print()
print(centroids)
plt.subplot(121),plt.imshow(im_with_keypoints)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(labels)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()