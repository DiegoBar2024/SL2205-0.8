## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
from sklearn.cluster import KMeans
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

## -------------------------------------------- CLUSTERING ----------------------------------------------

## Especifico el ID de la persona que voy a usar para el estudio
id_persona = 256

## La idea es poder guardar todos los escalogramas generados como imágenes con extensión .png en una carpeta aparte
## Especifico la ruta de donde voy a leer los escalogramas
ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo_gp/S{}/".format(id_persona)

## Especifico la ruta de donde voy a guardar los escalogramas
ruta_guardado = "C:/Yo/Tesis/sereData/sereData/Escalogramas_gp/S{}/".format(id_persona)

## En caso de que el directorio no exista
if not os.path.exists(ruta_guardado):

    ## Creo el directorio correspondiente
    os.makedirs(ruta_guardado)

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

## Aplico un clustering KMeans a los datos correspondientes de entrada
kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = "auto").fit(datos_entrada)

## Obtengo un vector con las etiquetas correspondientes a cada clase
## Voy a tener como resultado una etiqueta por cada punto de entrada, indicando el cluster
etiquetas = kmeans.labels_