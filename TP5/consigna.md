# Sistemas de Inteligencia Artificial: TP5: Deep Learning

## 1. Autoencoders

### a) Implementar un Autoencoder básico para las imágenes binarias de la lista de caracteres del archivo "font.h"

1. Plantear una arquitectura de red para el Codificador y Decodificador que permita representar los datos de entrada en un espacio latente de dos dimensiones.
2. Describan y estudien las diferentes arquitecturas y técnicas de optimización que fueron aplicando para permitir que la red aprenda todo el set de datos o un subconjunto del mismo. El objetivo es aprender los 32 patrones de $5 \times 7$ en un espacio latente de 2 dimensiones con un error máximo de 1 pixel incorrecto. En el caso de que sea un subconjunto, mostrar por qué no fue posible aprender el dataset completo.
3. Realizar el gráfico en dos dimensiones que muestre los datos de entrada en el espacio latente.
4. Mostrar cómo la red puede generar una nueva letra que no pertenece al conjunto de entrenamiento.

### b) Sobre el mismo dataset, implementar una variante que implemente un "Denoising Autoencoder".

1. Plantear una arquitectura de red conveniente para esta tarea. Explicar la elección.
2. Distorsionen las entradas en diferentes niveles y estudien la capacidad del Autoencoder de eliminar el ruido.

---

## 2. Extender el Autoencoder anterior para que opere como un VAE, un Autoencoder Variacional.

* **a)** Elegir (o construir) un conjunto de datos nuevos (por ejemplo emojis).
* **b)** Modificar el autoencoder planteando un esquema variacional para poder solucionar el problema de la representación en el espacio latente.
* **c)** Utilizar el Autoencoder anterior para generar una nueva muestra que ustedes puedan juzgar que pertenece al conjunto de datos que le presentaron al autoencoder (Generative Autoencoder).
