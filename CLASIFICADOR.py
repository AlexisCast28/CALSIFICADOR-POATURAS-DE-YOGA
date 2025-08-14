import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

TI = (320, 320)
MODEL_PATH = "MODELO_YOGA.h5"
TEST_FOLDER = r"C:\Users\tron_\Desktop\VISION ARTIFICIAL\PROYECTO UNIDAD 3\MODELO_YOGA_CPU.h5""

model = load_model(MODEL_PATH)
X_TEST = []
Y_PATHS = []

for class_name in os.listdir(TEST_FOLDER):
    class_folder = os.path.join(TEST_FOLDER, class_name)
    if not os.path.isdir(class_folder):
        continue
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, TI)
        img = img.astype("float32") / 255.0
        X_TEST.append(img)
        Y_PATHS.append(class_name)

X_TEST = np.array(X_TEST)
preds = model.predict(X_TEST)

for i, pred in enumerate(preds):
    clase_idx = np.argmax(pred)
    print("IMAGEN: " + Y_PATHS[i] + " -> CLASE PREDICHA: " + str(clase_idx))
