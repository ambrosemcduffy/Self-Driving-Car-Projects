import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Importing the image
image = mpimg.imread('street.jpg')
# Creating a copy
image_copy = image.copy()
plt.figure(figsize=(20, 20))
# Creating an image in HSV space
hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Creating low, and high threshold for inRange Function
lower_yellow = np.array([8, 33, 205])
upper_yellow = np.array([28, 53, 285])

# Creating Hsv Mask
hsv_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
hsv_mask = cv2.GaussianBlur(hsv_mask, (3, 3), 0)

# Keying out the background
target = cv2.bitwise_and(image, image, mask=hsv_mask)
# Obtaining the edges
edges = cv2.Canny(target, 1, 10)

# Creating an empty mask
mask = np.zeros_like(edges)

line_segment = []
imshape = image.shape

# Setting vertices for polyshape
vertices = np.array([[(5, imshape[0]),
                      (750, 350), (790, 370),
                      (800, imshape[0])]], dtype=np.int32)
poly = cv2.fillPoly(mask, vertices, 255)

masked_edges = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(masked_edges,
                        2,
                        np.pi/180,
                        100,
                        np.array([]),
                        100, 300)

line_left = []
line_right = []
right_slope = []
left_slope = []

for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2-y1)/(x2-x1)
        if slope > 0.0:
            right_slope.append(slope)
            line_right.append([[x1, y1, x2, y2]])
        elif slope < 0.0:
            if not np.isinf(slope):
                left_slope.append(slope)
                line_left.append([[x1, y1, x2, y2]])


def extract_lines(lane):
    start_x = []
    start_y = []
    end_x = []
    end_y = []
    for line in lane:
        for x1, y1, x2, y2 in line:
            start_x.append(x1)
            start_y.append(y1)
            end_x.append(x2)
            end_y.append(y2)
    return start_x+end_x, start_y+end_y


def new_line(lane, threshold):
    x, y = extract_lines(lane)
    line = np.polyfit(x, y, 1)
    m = line[0]
    b = line[1]
    maxY = imshape[0]
    y1 = maxY
    x1 = int((y1-b)/m)
    y2 = int((maxY/2)+threshold)
    x2 = int((y2-b)/m)
    return (x1, y1, x2, y2)


def show_line(line, color, threshold):
    (x1, y1, x2, y2) = new_line(line, threshold)
    cv2.line(image_copy,
             (x1, y1),
             (x2, y2),
             color,
             15)
    plt.imshow(image_copy)


thresh = 100
show_line(line_right, (255, 0, 0), thresh)
show_line(line_left, (0, 0, 255), thresh)
