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

import os
os.makedirs('images', exist_ok=True)
cv2.imwrite('images/marker_0_cube.png', img_0)
cv2.imwrite('images/marker_1_bowl.png', img_1)

print("Successfully generated images/marker_0_cube.png and images/marker_1_bowl.png")
