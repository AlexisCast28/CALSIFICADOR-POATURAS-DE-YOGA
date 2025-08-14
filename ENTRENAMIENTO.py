import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

X_TRAIN = np.load("X_TRAIN.npy")
Y_TRAIN = np.load("Y_TRAIN.npy")
X_TEST = np.load("X_TEST.npy")
Y_TEST = np.load("Y_TEST.npy")

NUM_CLASES = len(np.unique(Y_TRAIN))
Y_TRAIN = to_categorical(Y_TRAIN, NUM_CLASES)
Y_TEST = to_categorical(Y_TEST, NUM_CLASES)

MODELO = Sequential()
MODELO.add(Conv2D(16, (3,3), activation='relu', input_shape=(320,320,3)))
MODELO.add(MaxPooling2D((2,2)))
MODELO.add(Conv2D(32, (3,3), activation='relu'))
MODELO.add(MaxPooling2D((2,2)))
MODELO.add(Flatten())
MODELO.add(Dense(64, activation='relu'))
MODELO.add(Dense(NUM_CLASES, activation='softmax'))

MODELO.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

MODELO.fit(X_TRAIN, Y_TRAIN, epochs=10, batch_size=8, validation_data=(X_TEST, Y_TEST))

MODELO.save("MODELO_YOGA_CPU.h5")
