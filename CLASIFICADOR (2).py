import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

TI = (320, 320)
MODEL_PATH = r"C:\Users\tron_\Desktop\VISION ARTIFICIAL\PROYECTO UNIDAD 3\MODELOS\MODELO_YOGA_CPU.h5"
TEST_FOLDER = r"C:\Users\tron_\Desktop\VISION ARTIFICIAL\PROYECTO UNIDAD 3\DATASET\TEST_FINAL"

POSICIONES = ["COBRA POSE", "DOWNWARDOG", "GODDESS", "PLANK", "TREE POSSE", "WARRIOR2"]  # Ajusta según tu modelo

model = load_model(MODEL_PATH)

resultados = []

# Recorre todas las subcarpetas e imágenes
for root, dirs, files in os.walk(TEST_FOLDER):
    for img_name in files:
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(root, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue
        img = cv2.resize(img, TI)
        img = img.astype("float32") / 255.0
        img_array = np.expand_dims(img, axis=0)
        pred = model.predict(img_array)
        clase_idx = np.argmax(pred, axis=1)[0]
        clase_nombre = POSICIONES[clase_idx] if clase_idx < len(POSICIONES) else str(clase_idx)
        resultados.append(f"{img_name}: {clase_nombre}")
        print(f"{img_name} -> {clase_nombre}")

# Guarda todo al final en el .txt
with open("resultados.txt", "w") as f:
    for linea in resultados:
        f.write(linea + "\n")

print("Clasificación completada, resultados guardados en resultados.txt")
