# -*- coding: utf-8 -*-
"""
Created on Wed May 28 23:22:10 2025

@author: diego
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0) 

while 1:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo capturar el video.")
        break

    # Convertir a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir el rango de color (aquí está configurado entre verde claro y naranja)
    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    # Crear máscara binaria
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Extraer color usando la máscara
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Definir el kernel para operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)

    # Operaciones morfológicas
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Top-Hat: resalta objetos pequeños brillantes en la máscara
    tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)

    # Black-Hat: resalta huecos oscuros pequeños dentro de regiones brillantes
    blackhat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)

    # Mostrar resultados
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)
    cv2.imshow('Top-Hat', tophat)
    cv2.imshow('Black-Hat', blackhat)
    cv2.imshow('Color Detectado', res)

    # Salir con tecla ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
