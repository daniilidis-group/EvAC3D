import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
import torch

''' 
take a binary mask and get the contours
'''
def get_contour(binary_image):
    binary_image = (255*binary_image).astype(np.uint8)
    contours, hierarchy = cv2.findContours(image=binary_image, 
                                    mode=cv2.RETR_TREE, 
                                    method=cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        contours = [contours[0]]
        return contours
    return []

def draw_contours(image, contours):
    if len(contours) > 0:
        cv2.drawContours(image=image, 
                            contours=contours, 
                            contourIdx=-1, color=(0, 0, 255), 
                            thickness=2, 
                            lineType=cv2.LINE_AA)
    return image

# sample uniformly from a circle
# return shape: [2, n_points]
def sample_circle(n_points=1000):
    r = 1
    theta = np.linspace(0, 2*np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return torch.tensor(np.stack((x, y), axis=0))
