# -*- coding: utf-8 -*-
"""
Created on Fri May 16 

@author: diego
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar imagen
img = cv2.imread('Ferrari812.jpg', cv2.IMREAD_GRAYSCALE)

# Ecualización
img_eq = cv2.equalizeHist(img)
    
# Histogramas de intensidad
histx1 = cv2.calcHist([img], [0], None, [256], [0, 256])
histx2 = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# Histogramas verticales (suma por columna)
histy1 = np.sum(img, axis=0)
histy2 = np.sum(img_eq, axis=0)

# Plot
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(3, 4)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(histx1, color='black')
ax1.set_title('Histograma - Original')
ax1.set_xlim([0, 256])

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(histx2, color='black')
ax2.set_title('Histograma - Ecualizada')
ax2.set_xlim([0, 256])

ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(img, cmap='gray')
ax4.set_title('Imagen Original')
ax4.axis('off')


ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(img_eq, cmap='gray')
ax6.set_title('Imagen Ecualizada')
ax6.axis('off')



plt.tight_layout()
plt.show()