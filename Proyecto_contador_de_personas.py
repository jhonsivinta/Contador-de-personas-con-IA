import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random
from ipywidgets import Video, Image
from IPython.display import display
import base64
from PIL import Image
from matplotlib import pyplot as plt

# Abrir el archivo de video
cap = cv2.VideoCapture('almacen.mp4')

# Especificación 1: Sistema de detección robusto y preciso
videopath = "almacen.mp4"
model = YOLO("models/yolov8n.pt", task="detect")
colors = random.choices(range(256), k=1000)


# Crear objeto sustractor de fondo
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Lista para almacenar las coordenadas de las personas previamente detectadas
personas_previas = []

# Función para mostrar texto en una ventana emergente
def mostrar_texto_emergente(frame, texto, posicion):
    cv2.rectangle(frame, (10, posicion), (500, posicion + 50), (0, 0, 0), -1)  # Cuadro de fondo
    cv2.putText(frame, texto, (20, posicion + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Texto

# Bucle principal
while True:
    # Leer el fotograma del video
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Definir un área de interés (región de conteo)
    area_pts = np.array([[600, 30], [300, 30], [530, frame.shape[1]], [930, frame.shape[1]]])

    # Crear una máscara para el área de interés
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    # Aplicar sustracción de fondo y operaciones morfológicas para detectar movimiento
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Encontrar contornos en la máscara de movimiento
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    personas_detectadas = []

    # Identificar y dibujar contornos de personas detectadas
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            personas_detectadas.append((x, y, x+w, y+h))

    # Verificar y actualizar las personas previas con las nuevas detecciones
    personas_nuevas = []
    for persona in personas_detectadas:
        x1, y1, x2, y2 = persona
        nueva_persona = True

        for previa in personas_previas:
            px1, py1, px2, py2 = previa
            overlap_area = (min(x2, px2) - max(x1, px1)) * (min(y2, py2) - max(y1, py1))
            area_persona = (x2 - x1) * (y2 - y1)
            area_previa = (px2 - px1) * (py2 - py1)
            overlap_coefficient = overlap_area / min(area_persona, area_previa)

            if overlap_coefficient > 0.5:
                nueva_persona = False
                break

        if nueva_persona:
            personas_nuevas.append(persona)

    personas_previas = personas_nuevas.copy()

    # Mostrar el conteo de personas en una ventana emergente
    texto_conteo = f"Personas: {len(personas_previas)}"
    mostrar_texto_emergente(frame, texto_conteo, 30)

    # Mostrar información de estado en una ventana emergente
    texto_estado = "No se ha detectado movimiento" if not personas_detectadas else "Alerta Movimiento Detectado!"
    color = (0, 255, 0) if not personas_detectadas else (0, 0, 255)
    mostrar_texto_emergente(frame, texto_estado, 70)

    # Dibujar área de interés en el fotograma
    cv2.drawContours(frame, [area_pts], -1, color, 2)

    # Mostrar las imágenes procesadas en ventanas separadas
    #cv2.imshow('fgmask', fgmask)
    cv2.imshow("frame", frame)

    # Esperar la pulsación de la tecla 'ESC' para salir del bucle
    k = cv2.waitKey(70)
    if k == 27 or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:  # Salir si se presiona 'ESC' o se cierra la ventana
        break

# Liberar los recursos y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
