import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 'n' = nano, el más ligero

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección
    results = model.predict(source=frame, conf=0.5, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordenadas de la caja
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Dibujar caja y etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow('Detección de objetos - YOLOv8', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
