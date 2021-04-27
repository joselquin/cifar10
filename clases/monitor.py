# Monitor - Detección de entrenamiento con overfitting
# Dibuja un gráfico con los valores de loss y accuracy tanto para en dataset de entrenamiento como para el de 
# validación después de cada epoch. Si los valores de entrenamiento y validación empiezan a diverger demasiado, 
# tendremos una indicación de overfitting. 
# Esta clase se puede usar como parte de un callback de Keras.

from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class Monitor(BaseLogger):
    def __init__(self, ruta_plot, ruta_json=None, comienzo=0):
        # ruta_plot - Ruta donde se graba el plot de resultados de loss/accuracy de entrenamiento y validación
        # ruta_json - Ruta donde se graba un json con datos de loss/accuracy
        # comienzo - Epoch donde comienza la monitorización
        super(Monitor, self).__init__()
        self.ruta_plot = ruta_plot
        self.ruta_json = ruta_json
        self.comienzo = comienzo

    def on_train_begin(self, logs={}):
        # Diccionario donde se graba el historial con los datos de loss y accuracy
        self.H = {}

        # Carga el historial si existe el json
        if self.ruta_json is not None:
            if os.path.exists(self.ruta_json):
                self.H = json.loads(open(self.ruta_json).read())

                # Si hay variable comienzo suministrada, borra del historial lo que haya grabado después del epoch suministrado
                if self.comienzo > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.comienzo]

    def on_epoch_end(self, epoch, logs={}):
        # Actualiza el historial de loss y accuracy al final de cada epoch en el diccionario
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # Graba historial a un archivo
        if self.ruta_json is not None:
            f = open(self.ruta_json, "w")
            f.write(json.dumps(self.H))
            f.close()

        # Crea el plot con los loss y accuracy
        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss & Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # Graba el plot
            plt.savefig(self.ruta_plot)
            plt.close()
