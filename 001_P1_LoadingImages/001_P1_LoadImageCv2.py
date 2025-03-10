# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 19:56:47 2025

@author: diego
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Ferrari812.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Ferrari812Gray.png', img)