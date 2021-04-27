import matplotlib
matplotlib.use("Agg")

from clases.minivggnet import MiniVGGNet
from clases.monitor import Monitor
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np 
import argparse, os

epochs = 40
batch_size = 64

# Argumentos a indicar en terminal:
# output - Ruta donde se graba el plot de resultados de loss/accuracy de entrenamiento y validación,
#          el json con el historial de loss/accuracy y los pesos tras cada epoch
# comienzo - epoch donde comienza la monitorización
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Ruta al plot de loss/accuracy, json con historial y los pesos")
ap.add_argument("-c", "--comienzo", required=False, help="Epoch donde comienza la monitorización")
args = vars(ap.parse_args())
ruta_plot = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
ruta_json = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
ruta_modelo = os.path.sep.join([args["output"], "{}.hdf5".format(os.getpid())])

print("***** Cargando CIFAR-10...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# Normaliza las imágenes tanto en el trainset como en el testset
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# OneHotEncoding con las etiquetas
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)
labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Definición del modelo, usando la clase MiniVGGNet
opt = SGD(lr=0.01, decay=0.01 / epochs, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Crea un callback para la monitorización del entrenamiento
monitor = [Monitor(ruta_plot=ruta_plot, ruta_json=ruta_json)]

# Crea otro callback para la grabación del modelo solo cuando se produce una mejora de accuracy en 
# el dataset de validación
checkpoint = ModelCheckpoint(ruta_modelo, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
callbacks = [monitor, checkpoint]

print("**** Training...")
H = model.fit(trainX, trainY, 
    validation_data=(testX, testY), 
    batch_size=batch_size, 
    epochs=epochs, 
    callbacks=[callbacks],
    verbose=1)

print("**** Evaluating...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labels))


