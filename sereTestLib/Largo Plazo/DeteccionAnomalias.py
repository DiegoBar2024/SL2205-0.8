## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatos import *
from Muestreo import *
from LecturaDatosPacientes import *
from DeteccionActividades import DeteccionActividades
from joblib import load
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

## -------------------------------- DISCRIMINACIÓN REPOSO - ACTIVIDAD --------------------------------------

## La idea de ésta parte consiste en poder hacer una discriminación entre reposo y actividad
## Especifico la ruta en la cual se encuentra el registro a leer
ruta_registro = 'C:/Yo/Tesis/sereData/sereData/Registros/Actividades_Rodrigo.txt'

##  Hago la lectura de los datos
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = None, lectura_datos_propios = True, ruta = ruta_registro)

## Defino la cantidad de muestras de la ventana que voy a tomar
muestras_ventana = 200

## Defino la cantidad de muestras de solapamiento entre ventanas
muestras_solapamiento = 100

## Hago el cálculo del vector de SMA para dicha persona
vector_SMA, features = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras, actividad = None)

## Cargo el modelo del clasificador ya entrenado según la ruta del clasificador
clf_entrenado = load("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Largo Plazo/SVM.joblib")

## Determino la predicción del clasificador ante mi muestra de entrada
## Etiqueta 0: Reposo
## Etiqueta 1: Movimiento
pat_predictions = clf_entrenado.predict(np.array((vector_SMA)).reshape(-1, 1))

## ------------------------------------------ SEGMENTACIÓN ---------------------------------------------

## La idea en ésta parte es poder segmentar las señales en reposo y movimiento
## Genero una lista donde voy a guardar las posiciones de los segmentos clasificados de igual manera
tramos_actividades = [[0, 0]]

## Itero para cada uno de los índices predichos
for i in range (len(pat_predictions) - 1):

    ## En caso de que haya una transición de movimiento a reposo
    if pat_predictions[i] != pat_predictions[i + 1]:

        ## Me guardo los índices correspondientes a donde la actividad es igual
        tramos_actividades.append([tramos_actividades[-1][1] + 1, i])

## Elimino el dummy vector del inicio y lo transformo en un vector numpy
## Obtengo una matriz donde:
## La i-ésima fila hace referencia al i-ésimo segmento uniforme
## La columna 0 es el indice de la posición inicial, la columna 1 es el indice de posicion final
tramos_actividades = np.array((tramos_actividades[1:]))

## ------------------------------------------- ANOMALÍAS ----------------------------------------------

## Hago la conversión de los features a un array bidimensional
## Obtengo una matriz donde:
## La i-ésima fila hace referencia al i-ésimo segmento
## La j-ésima columna hace referencia al j-ésimo feature
features = np.array((features))

## Hago la normalización de la matriz de features
features_norm = normalize(features, norm = "l2", axis = 0)

## Especifico una lista con la cantidad de clusters que voy a usar durante el análisis
clusters = np.linspace(2, 10, 9).astype(int)

## Itero para cada una de los tramos que tengo detectados
for i in range (tramos_actividades.shape[0]):

    ## Construyo un vector en donde voy a guardar los silhouette scores para cada una de las distribuciones de clusters
    silhouette_scores = []

    ## Itero para cada una de las cantidades de clusters que tengo
    for nro_clusters in clusters:

        print(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]].shape[0])
    
        ## En caso de que el número de clusters en la iteración sea mayor a la cantidad de puntos en el dataset
        if nro_clusters > features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]].shape[0] - 1:

            continue

        ## Aplico un clustering KMeans a los datos correspondientes de entrada
        kmeans = KMeans(n_clusters = nro_clusters, random_state = 0, n_init = "auto").fit(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

        ## Agrego la silhouette score a la lista de indicadores
        silhouette_scores.append(silhouette_score(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]], kmeans.fit_predict(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])))

    ## En caso de que no haya silhouette scores
    if len(silhouette_scores) == 0:

        ## Asigno 1 como la cantidad óptima de clústers
        clusters_optimo = 1
    
    ## En caso de que tenga silhouette scores
    else:

        ## Obtengo la cantidad óptima de clústers observando aquel número en donde se maximice la silhouette score
        clusters_optimo = np.argmax(silhouette_scores) + 2

    ## Aplico el clústering KMeans pasando como entrada el número óptimo de clústers determinado por el Silhouette Score
    kmeans = KMeans(n_clusters = clusters_optimo, random_state = 0, n_init = "auto").fit(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

    ## Construyo un vector donde voy a guardar la distancia euclideana de cada punto a su respectivo centroide
    distancias_puntos = []

    ## Itero para cada uno de los puntos del dataset
    for j in range (len(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])):

        ## Obtengo el centroide del clúster que está asociado al i-ésimo punto
        centroide = kmeans.cluster_centers_[kmeans.labels_[j]]

        ## Calculo la distancia entre el i-ésimo punto y el centroide asociado (usando norma Euclideana)
        distancia = np.linalg.norm(centroide - features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]][j])

        ## Agrego la distancia del punto a su centroide a la lista
        distancias_puntos.append(distancia)