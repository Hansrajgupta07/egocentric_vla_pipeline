import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

try:
    # Newer OpenCV versions
    img_0 = cv2.aruco.generateImageMarker(aruco_dict, 0, 400)
    img_1 = cv2.aruco.generateImageMarker(aruco_dict, 1, 400)
except AttributeError:
    # Older OpenCV versions
    img_0 = np.zeros((400, 400), dtype=np.uint8)
    img_1 = np.zeros((400, 400), dtype=np.uint8)
    img_0 = cv2.aruco.drawMarker(aruco_dict, 0, 400, img_0, 1)
    img_1 = cv2.aruco.drawMarker(aruco_dict, 1, 400, img_1, 1)

cv2.imwrite('marker_0_cube.png', img_0)
cv2.imwrite('marker_1_bowl.png', img_1)

print("Successfully generated marker_0_cube.png and marker_1_bowl.png")
