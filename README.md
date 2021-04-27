# cifar10
CNN aplicada a Cifar10 con monitorización de overfitting y grabación de modelo óptimo.

Se usa la arquitectura VGGNet (light) y dos callbacks muy útiles:

- Monitorización, dibujando después de cada epoch los valores de loss y accuracy de los datasets de entrenamiento y validación. Así, podemos vigilar tras cada epoch si hay divergencia preocupante y detener el proceso si se presenta overfitting.
- Checkpoint, usando el callback ModelCheckpoint de Keras para grabar después de cada epoch solo los pesos cuando la accuracy aumente, descartando los epochs donde no lo haga.
