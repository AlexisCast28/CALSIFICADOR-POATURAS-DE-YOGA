import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

TI = (255, 255)
DATA = r"C:\Users\tron_\Desktop\VISION ARTIFICIAL\PROYECTO UNIDAD 3\DATASET"

def LOAD_DATASET(PATH):
    X = []
    Y = []
    POSICIONES = os.listdir(PATH)

    for IDX, CLASS_NAME in enumerate(POSICIONES):
        CLASS_FOLDER = os.path.join(PATH, CLASS_NAME)
        if not os.path.isdir(CLASS_FOLDER):
            continue

        print("PROCESANDO CLASE: " + CLASS_NAME + " EN " + PATH)

        for IMG_NAME in os.listdir(CLASS_FOLDER):
            IMG_PATH = os.path.join(CLASS_FOLDER, IMG_NAME)
            IMG = cv2.imread(IMG_PATH)

            if IMG is None or IMG.size == 0:
                print("NO SE PUEDE LEER: " + IMG_PATH)
                continue

            IMG = cv2.resize(IMG, TI)
            IMG = IMG.astype("float32") / 255.0

            X.append(IMG)
            Y.append(IDX)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y, POSICIONES

if __name__ == "__main__":
    TRAIN = os.path.join(DATA, "TRAIN")
    TEST = os.path.join(DATA, "TEST")

    print("CARGANDO DATASET...")

    X_TRAIN, Y_TRAIN, TRAIN_POS = LOAD_DATASET(TRAIN)
    X_TEST, Y_TEST, TEST_POS = LOAD_DATASET(TEST)

    print("TOTAL IMAGENES TRAIN: " + str(len(X_TRAIN)))
    print("TOTAL IMAGENES TEST: " + str(len(X_TEST)))
    print("CLASES TRAIN: " + str(TRAIN_POS))
    print("CLASES TEST: " + str(TEST_POS))

    np.save("X_TRAIN.npy", X_TRAIN)
    np.save("Y_TRAIN.npy", Y_TRAIN)
    np.save("X_TEST.npy", X_TEST)
    np.save("Y_TEST.npy", Y_TEST)
# MOSTRAR ALGUNAS IMAGENES DE TRAIN
for i in range(5):
    plt.imshow(X_TRAIN[i])
    plt.title("CLASE: " + TRAIN_POS[Y_TRAIN[i]])
    plt.show()

# MOSTRAR ALGUNAS IMAGENES DE TEST
for i in range(5):
    plt.imshow(X_TEST[i])
    plt.title("CLASE: " + TEST_POS[Y_TEST[i]])
    plt.show()


    print("DATOS PROCESADOS Y GUARDADOS.")
