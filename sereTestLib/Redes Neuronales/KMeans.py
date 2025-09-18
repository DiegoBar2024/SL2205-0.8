## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import keras
import sys
from scipy.signal import *
import scipy.spatial
import numpy as np
from sklearn.cluster import KMeans
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatosPacientes import *
from scipy.stats import *
from sklearn.datasets import make_classification

## ------------------------------------------ KMEANS CASERO --------------------------------------------

## Genero un dataset aleatorio
X, y = make_classification(n_samples = 1000, n_features = 256)

## Especifico la cantidad de clústers que tengo
nro_clusters = 3

## Elijo un conjunto de indices aleatorios para las filas
indices_aleatorios = np.random.choice(10, size = nro_clusters, replace = False)

## Obtengo los centroides iniciales de los clústers
## La i-ésima fila va a ser el i-ésimo centroide
## La j-ésima columna va a ser el j-ésimo centroide de dicho feature
centroides = X[indices_aleatorios]

## Inicializo una variable donde guardo las iteraciones
iteraciones = 0

## Mientras no haya convergencia, sigo iterando
while True:

    ## Sumo una unidad a las iteraciones
    iteraciones += 1

    ## Construyo un diccionario donde la clave sea el numero de cluster
    clusters = {}

    ## Inicializo el diccionario colocando los números de cluster
    for i in range(0, nro_clusters):

        ## Configuro la clave
        clusters[i] = []

    ## -------- ASIGNACIÓN DE LOS PUNTOS AL CLÚSTER MÁS PRÓXIMO -------

    ## Itero para cada uno de los puntos que tengo en el dataset
    for x in X:

        ## Genero una lista de distancias del punto a los clusters
        distancia_centroide = []

        ## Itero para cada uno de los centroides que tengo
        for c in range (nro_clusters):

            ## Calculo la distancia del punto x al centroide c
            distancia_centroide.append(np.linalg.norm(x - centroides[c, :]))
        
        ## Obtengo el índice del clúster cuyo centroide sea el más próximo
        cluster_asignado = np.argmin(np.array(distancia_centroide))

        ## Agrego el punto al diccionario de clusters
        clusters[cluster_asignado].append(x)

    ## -------- ACTUALIZACIÓN DE CENTROIDES DE LOS CLÚSTERS -------

    ## Seteo una bandera que me diga que los centroides han cambiado
    dejar_de_iterar = True

    ## Itero para cada uno de los clústers
    for c in range(nro_clusters):

        ## En caso de que haya un centroide distinto al anterior
        if np.linalg.norm(centroides[c, :] - np.mean(np.array(clusters[c]), axis = 0)) != 0:

            ## Sigo iterando
            dejar_de_iterar = False

        ## Actualizo el centroide calculando el vector medio de los puntos asignados
        centroides[c, :] = np.mean(np.array(clusters[c]), axis = 0)

        print(np.linalg.norm(centroides[c, :] - np.mean(np.array(clusters[c]))))
    
    ## En caso de que diga que deje de iterar
    if dejar_de_iterar:

        ## Termino el bucle while
        break

    print(iteraciones)

## Aplico el algoritmo K-Means
kmeans = KMeans(n_clusters = nro_clusters, random_state = 0, n_init = "auto").fit(X)