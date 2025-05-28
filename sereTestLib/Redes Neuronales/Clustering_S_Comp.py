## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
from sklearn.cluster import KMeans
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

## ----------------------------------------- CARGADO DE DATOS -------------------------------------------

## Especifico el ID de la persona que voy a usar para el estudio
id_persona = 256

## La idea es poder guardar todos los escalogramas generados como imágenes con extensión .png en una carpeta aparte
## Especifico la ruta de donde voy a leer los escalogramas
ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/S{}/".format(id_persona)

## Obtengo todos los archivos presentes en la ruta anterior
## Recuerdo que cada uno de los archivos se corresponde con un escalograma (tensor tridimensional)
archivos = [archivo for archivo in os.listdir(ruta_lectura) if archivo.endswith("npz")]

## Construyo una matriz donde las filas son las observaciones y las columnas los features
datos_entrada = np.zeros((len(archivos), 6 * 128 * 800))

## Itero para todos los archivos presentes en el directorio
for i in range (len(archivos)):

    ## Abro el archivo con la extensión .npz donde se encuentra el escalograma
    archivo_escalograma = np.load(ruta_lectura + archivos[i])

    ## Almaceno el escalograma correspondiente en la variable <<escalograma>>
    escalograma = archivo_escalograma['X']

    ## Hago un flatten para crear un vector en base al escalograma
    vector_escalograma = np.ndarray.flatten(escalograma)

    ## Agrego el vector conteniendo el escalograma a la matriz correspondiente
    datos_entrada[i, :] = vector_escalograma

## ------------------------------------------- CLUSTERING -------------------------------------------------

## Especifico una lista con la cantidad de clusters que voy a usar durante el análisis
clusters = np.linspace(1, 20, 20).astype(int)

## Creo un vector en donde me guardo la distorsion score correspondiente al número de clusters
distorsion_scores = []

## Itero para cada una de las cantidades de clusters que tengo
for nro_clusters in clusters:

    ## Aplico un clustering KMeans a los datos correspondientes de entrada
    kmeans = KMeans(n_clusters = nro_clusters, random_state = 0, n_init = "auto").fit(datos_entrada)

    ## Obtengo el distorsion score correspondiente
    distorsion_score = kmeans.inertia_

    ## Me lo guardo en el vector
    distorsion_scores.append(distorsion_score)

plt.plot(clusters, distorsion_scores)
plt.show()