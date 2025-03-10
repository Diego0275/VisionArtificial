# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:14:20 2025

@author: diego
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Ferrari812.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()