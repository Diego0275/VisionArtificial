# -*- coding: utf-8 -*-
"""
Created on Fri May 16 

@author: diego
"""

import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('Ferrari812.jpg', cv2.IMREAD_COLOR)

# Verificar carga
if img is None:
    print("Error al cargar la imagen.")
    exit()

# Tamaño original de la imagen
img_height, img_width = img.shape[:2]

# Extraer una región (ROI) de ejemplo
# Elegimos una zona razonable: (x1=200, y1=100) a (x2=400, y2=250)
roi = img[100:250, 200:400]  # ROI de 150x200

# Redimensionar la ROI (duplicar tamaño)
roi_resized = cv2.resize(roi, (400, 300))  # Nueva ROI de 300x400

# Calcular el centro donde pegarlo
center_x = img_width // 2
center_y = img_height // 2

# Coordenadas superiores izquierdas para pegar la ROI centrada
x_offset = center_x - roi_resized.shape[1] // 2
y_offset = center_y - roi_resized.shape[0] // 2

# Pegar la ROI redimensionada en el centro
img[y_offset:y_offset+roi_resized.shape[0], x_offset:x_offset+roi_resized.shape[1]] = roi_resized

# Mostrar información de la imagen
print("Tamaño:", img.shape)
print("Tamaño de ROI pegada:", roi_resized.shape)

# Mostrar imagen
cv2.imshow('Imagen modificada', img)
cv2.waitKey(0)
cv2.destroyAllWindows()