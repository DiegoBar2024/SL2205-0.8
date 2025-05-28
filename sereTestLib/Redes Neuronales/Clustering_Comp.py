## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
from sklearn.cluster import KMeans
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatosPacientes import *

## ----------------------------------- CARGADO DE DATOS COMPRIMIDOS ----------------------------------------------

## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
pacientes, ids_existentes = LecturaDatosPacientes()

## Genero un vector vacío para poder concatenar los vectores con todas las representaciones latentes
comprimidos_total = np.zeros((1, 256))

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Especifico la ruta de donde voy a leer los escalogramas comprimidos
        ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/latente_ae/S{}/".format(id_persona)

        ## Hago la lectura del archivo con el espacio latente
        archivo_comprimido = np.load(ruta_lectura + 'S{}_latente.npz'.format(id_persona))

        ## Almaceno el archivo comprimido en una variable como un array bidimensional
        ## La i-ésima fila representa el i-ésimo segmento
        ## La j-ésima columna representa el j-ésimo feature
        espacio_comprimido = archivo_comprimido['X']

        ## Concateno el espacio comprimido por filas a la matriz donde guardo los espacios latentes totales
        comprimidos_total = np.concatenate((comprimidos_total, espacio_comprimido), axis = 0)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))

    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue

## -------------------------------------------- CLUSTERING ----------------------------------------------

## Especifico una lista con la cantidad de clusters que voy a usar durante el análisis
clusters = np.linspace(2, 30, 10).astype(int)

## Creo un vector en donde me guardo la distorsion score correspondiente al número de clusters
distorsion_scores = []

## Creo un vector en donde me guardo el silhouette score correspondiente al número de clusters
silhouette_scores = []

## Itero para cada una de las cantidades de clusters que tengo
for nro_clusters in clusters:

    ## Aplico un clustering KMeans a los datos correspondientes de entrada
    kmeans = KMeans(n_clusters = nro_clusters, random_state = 0, n_init = "auto").fit(comprimidos_total)

    ## Obtengo la distorsion score correspondiente al clustering realizado y lo guardo en el vector correspondiente
    distorsion_scores.append(kmeans.inertia_)

    ## Obtengo la silhouette score correspondiente al clustering realizado y lo guardo en el vector correspondiente
    silhouette_scores.append(silhouette_score(comprimidos_total, kmeans.labels_))

## Graficación de los distorsion scores en función del número de clusters
plt.plot(clusters, distorsion_scores)
plt.xlabel('Cantidad de Clústers')
plt.ylabel('Distortion Score')
plt.show()

## Graficación de los silhouette scores en función del numero de clusters
plt.plot(clusters, silhouette_scores)
plt.xlabel('Cantidad de Clústers')
plt.ylabel('Silhouette Score')
plt.show()