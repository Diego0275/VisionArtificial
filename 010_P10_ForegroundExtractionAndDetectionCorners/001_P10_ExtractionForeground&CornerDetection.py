import numpy as np
import cv2
from matplotlib import pyplot as plt

# Cargar imagen original
img = cv2.imread('Ferrari812.jpg')
if img is None:
    print("❌ Imagen no encontrada.")
    exit()

# Crear máscara inicial para grabCut
mask = np.zeros(img.shape[:2], np.uint8)

# Modelos internos para fondo y primer plano
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Rectángulo que rodea el objeto (ajusta según tu imagen)
rect = (150, 80, 650, 350)

# Aplicar grabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Generar máscara binaria final
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Aplicar máscara a la imagen (conserva color)
img_cut = img * mask2[:, :, np.newaxis]

# Mostrar con matplotlib para revisar resultado de grabCut
plt.imshow(img_cut)
plt.show()

# Convertir a escala de grises para detección de esquinas
gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)

# Asegurar que la imagen no esté vacía
if np.count_nonzero(gray) == 0:
    print("⚠️ La imagen resultante está vacía. Ajusta el rectángulo.")
    exit()

# Convertir a float32 como requiere goodFeaturesToTrack
gray_float = np.float32(gray)

# Detectar esquinas
corners = cv2.goodFeaturesToTrack(gray_float, 100, 0.01, 10)
if corners is not None:
    corners = corners.astype(np.intp)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_cut, (x, y), 4, (0, 255, 0), -1)
else:
    print("⚠️ No se detectaron esquinas.")

# Mostrar imagen final con esquinas marcadas
cv2.imshow('Esquinas Detectadas', img_cut)
cv2.waitKey(0)
cv2.destroyAllWindows()
