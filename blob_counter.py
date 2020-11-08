import cv2
import numpy as np
import sys

print('Opening image: ', sys.argv[1])


width = 1280
height = 720
road_width = int(width / 5 * 3)


# Create 1280x720 image holder for blobs
# black blank image
black_image = np.zeros(shape=[height, width, 1], dtype=np.uint8)

# Draw white exclusion zones
zoned_placeholder = cv2.rectangle(black_image, (0,0), ( (width - road_width) // 2, height), (255), -1)
zoned_placeholder = cv2.rectangle(black_image, (width-((width - road_width) // 2),0), (width, height), (255), -1)

# print(blank_image.shape)
cv2.imshow("Black Blank", zoned_placeholder)

# Read image
im = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
im = cv2.bitwise_not(im)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 750

# Filter by Circularity
params.filterByCircularity = False

# Filter by Convexity
params.filterByConvexity = False

# Filter by Inertia
params.filterByInertia = False

# Create detector with params 
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("Image", im)


# Calculate final score (sum all keypoint areas)
max_area = height * road_width
final_area = 0
for keypoint in keypoints :
    approx_area = ((keypoint.size / 2) ** 2) * np.pi
    final_area += approx_area

norm_score = max_area / final_area
if norm_score > 1 : norm_score = 1

print('Score: ', norm_score)

# Show and wait
cv2.waitKey(0)
