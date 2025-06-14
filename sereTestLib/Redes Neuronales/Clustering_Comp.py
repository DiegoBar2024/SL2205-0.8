## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import scipy.spatial
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatosPacientes import *

## ------------------------------------- SELECCIÓN DE MUESTRAS ----------------------------------------------

## Construyo una variable booleana de modo de poder filtrar aquellos pacientes etiquetados
filtrar_etiquetados = False

## Construyo una lista con todos aquellos pacientes denominados estables no añosos
id_estables_no_añosos = np.array([114, 127, 128, 129, 130, 133, 213, 224, 226, 44, 294])

## Construyo una lista con todos aquellos pacientes denominados estables añosos
## En principio estos pacientes se consideran como estables pero se van a mantener por separado del análisis
id_estables_añosos = np.array([67, 77, 111, 112, 115, 216, 229, 271, 273])

## Obtengo una lista con los identificadores de todos los pacientes estables
id_estables = np.concatenate([id_estables_añosos, id_estables_no_añosos])

## Construyo una lista con aquellos pacientes denominados inestables
id_inestables = np.array([69, 72, 90, 122, 137, 139, 142, 144, 148, 149, 158, 167, 178, 221, 223, 232, 256])

## Construyo una lista con los IDs de aquellos pacientes los cuales yo sé que están etiquetados
id_etiquetados = np.concatenate([id_estables, id_inestables])

## ----------------------------------- CARGADO DE DATOS COMPRIMIDOS ----------------------------------------------

## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
pacientes, ids_existentes = LecturaDatosPacientes()

## Creo un vector donde voy a guardar las etiquetas asociadas a cada uno de los pacientes etiquetados, respetando el orden
vector_etiquetas = []

## Genero un vector vacío para poder concatenar los vectores con todas las representaciones latentes
comprimidos_total = np.zeros((1, 256))

## Inicializo un diccionario cuya clave sea el ID del paciente y el valor una tupla con la posición inicial y final de los escalogramas
dicc_ID_pos = {}

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        if not id_persona == 114:

            continue

        ## Especifico la ruta de donde voy a leer los escalogramas comprimidos
        ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/latente_ae/S{}/".format(id_persona)

        ## Hago la lectura del archivo con el espacio latente
        archivo_comprimido = np.load(ruta_lectura + 'S{}_latente.npz'.format(id_persona))

        ## Almaceno el archivo comprimido en una variable como un array bidimensional
        ## La i-ésima fila representa el i-ésimo segmento
        ## La j-ésima columna representa el j-ésimo feature
        espacio_comprimido = archivo_comprimido['X']

        ## En caso de que yo quiera filtrar los pacientes etiquetados en mi muestra
        if filtrar_etiquetados:

            ## En caso de que el ID del paciente que está siendo analizado no corresponde a un paciente etiquetado, me lo salteo y no lo proceso
            if id_persona not in id_etiquetados:

                continue

            ## En caso de que el paciente haya sido clasificado como estable
            if id_persona in id_estables:

                ## Concateno un vector de ceros con la cantidad de segmentos que tengo para el registro del paciente
                vector_etiquetas = np.concatenate((np.array(vector_etiquetas), np.zeros((espacio_comprimido.shape[0])))).astype(int)
            
            ## En caso de que el paciente haya sido clasificado como inestable
            else:

                ## Concateno un vector de ceros con la cantidad de segmentos que tengo para el registro del paciente
                vector_etiquetas = np.concatenate((np.array(vector_etiquetas), np.ones((espacio_comprimido.shape[0])))).astype(int)

        ## Concateno el espacio comprimido por filas a la matriz donde guardo los espacios latentes totales
        comprimidos_total = np.concatenate((comprimidos_total, espacio_comprimido), axis = 0)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))

    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue

## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector)
## Se obtiene entonces una matriz de datos donde:
## La i-ésima fila representa el i-ésimo segmento
## La j-ésima columna representa el j-ésimo feature
comprimidos_total = comprimidos_total[1:,:]

## ------------------------------------ CANTIDAD ÓPTIMA DE CLÚSTERS ----------------------------------------------

## Normalización de la matriz de datos de entrada al algoritmo de clustering
## Hago la normalización por columna es decir por feature
comprimidos_total = normalize(comprimidos_total, norm = "l2", axis = 0)

## Especifico una lista con la cantidad de clusters que voy a usar durante el análisis
clusters = np.linspace(2, 30, 29).astype(int)

## Creo un vector en donde me guardo la inercia correspondiente al número de clusters
inercias = []

## Creo un vector en donde me guardo la distorsión correspondiente al numero de clusters
distorsiones = []

## Construyo un vector en donde voy a guardar los silhouette scores para cada una de las distribuciones de clusters
silhouette_scores = []

## Itero para cada una de las cantidades de clusters que tengo
for nro_clusters in clusters:

    ## Aplico un clustering KMeans a los datos correspondientes de entrada
    kmeans = KMeans(n_clusters = nro_clusters, random_state = 0, n_init = "auto").fit(comprimidos_total)

    ## ------------------------------ CÁLCULO DEL INTRA-CLUSTER ERROR ----------------------------------------------

    ## Inicializo la variable en la cual voy a guardar el error intra-cluster (inercia)
    error_intra_cluster = 0

    ## Itero para cada uno de los puntos del dataset
    for i in range (len(comprimidos_total)):

        ## Obtengo el centroide del clúster que está asociado al i-ésimo punto
        centroide = kmeans.cluster_centers_[kmeans.labels_[i]]

        ## Calculo la distancia entre el i-ésimo punto y el centroide asociado (usando norma Euclideana)
        distancia = np.linalg.norm(centroide - comprimidos_total[i])

        ## Sumo la distancia a la variable en donde guardo el error intra cluster
        error_intra_cluster += distancia ** 2

    ## ------------------------------ CÁLCULO DEL INTER-CLUSTER ERROR ----------------------------------------------

    ## Inicializo una variable en la cual voy a guardar el error inter-cluster
    error_inter_cluster = 0

    ## Inicializo una variable auxiliar que me guarde las distancias entre centroides
    distancias_centroides = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) ** 2

    ## Itero para cada uno de los centroides del algoritmo
    for i in range (len(kmeans.cluster_centers_)):

        ## Itero para los centroides a partir del actual (para no repetir pares)
        for j in range (i + 1, len(kmeans.cluster_centers_)):

            ## Obtengo la distancia entre los dos centroides
            distancia = np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j]) ** 2

            ## En caso de que sea menor a la distancia predefinida
            if distancia <= distancias_centroides:

                ## Actualizo el error inter cluster como la distancia entre este par de centroides
                error_inter_cluster = distancia
    
    ## ----------------------------------- CÁLCULO DE INDICADORES ------------------------------------------------

    ## Hago el cálculo del indicador que es el cociente entre el error inter-clase y el error intra-clase
    ## Yo quiero maximizar el error inter-clase y minimizar al mismo tiempo el error intra-clase
    indicador_error = error_inter_cluster / error_intra_cluster

    ## Obtengo la distorsión como el promedio de la inercia
    distorsion = kmeans.inertia_ / comprimidos_total.shape[0]

    ## Obtengo la inercia correspondiente al clustering realizado y lo guardo en el vector correspondiente
    inercias.append(kmeans.inertia_)

    ## Agrego la distrosión calculada a la lista de distorsiones
    distorsiones.append(distorsion)

    ## Agrego la silhouette score a la lista de indicadores
    silhouette_scores.append(silhouette_score(comprimidos_total, kmeans.fit_predict(comprimidos_total)))

# Grafico la inercia en función de la cantidad de clusters que tengo
plt.bar(clusters, silhouette_scores)
plt.xlabel("Numero de Clusters")
plt.ylabel("Silhouette Score")
plt.show()

## ------------------------------------------- CLUSTERIZADO ----------------------------------------------

## Obtengo la cantidad óptima de clústers observando aquel número en donde se maximice la silhouette score
clusters_optimo = np.argmax(silhouette_scores) + 2

## Aplico el clústering KMeans pasando como entrada el número óptimo de clústers determinado por el Silhouette Score
kmeans = KMeans(n_clusters = clusters_optimo, random_state = 0, n_init = "auto").fit(comprimidos_total)

## Construyo un vector donde voy a guardar la distancia euclideana de cada punto a su respectivo centroide
distancias_puntos = []

## Itero para cada uno de los puntos del dataset
for i in range (len(comprimidos_total)):

        ## Obtengo el centroide del clúster que está asociado al i-ésimo punto
        centroide = kmeans.cluster_centers_[kmeans.labels_[i]]

        ## Calculo la distancia entre el i-ésimo punto y el centroide asociado (usando norma Euclideana)
        distancia = np.linalg.norm(centroide - comprimidos_total[i])

        ## Agrego la distancia del punto a su centroide a la lista
        distancias_puntos.append(distancia)

## Graficación de las distancias de cada punto a los clusters
plt.scatter(np.linspace(0, len(distancias_puntos) - 1, len(distancias_puntos)), distancias_puntos)
plt.ylim(0, max(distancias_puntos) * 1.2)
plt.show()

## -------------------------------------------- TRAYECTORIA CLUSTERS ------------------------------------------------

## Obtengo los centroides correspondientes a cada clúster
centroides_clusters = kmeans.cluster_centers_

## Obtengo las etiquetas asociadas a los elementos del dataset según el cluster al que pertenecen (trayectoria)
etiquetas_cluster = kmeans.labels_

# ## Graficación de la trayectoria de los puntos a través de los diferentes clusters
# plt.scatter(np.linspace(0, len(etiquetas_cluster) - 1, len(etiquetas_cluster)), etiquetas_cluster)
# plt.show()

## ------------------------------------------- CLUSTERIZADO DISTANCIAS ----------------------------------------------

## Hago otra etapa de clusterizado pero para las distancias
## La idea es que al hacer clustering con K = 2 se puedan separar las muestras normales de las anormales por el criterio de distancias
kmeans_distancias = KMeans(n_clusters = 2, random_state = 0, n_init = "auto").fit(np.array((distancias_puntos)).reshape(-1, 1))

## ------------------------------------------- K-VECINOS MÁS PRÓXIMOS ------------------------------------------------

## Éste algoritmo implementa la distancia promedio entre un punto y sus K vecinos más próximos donde el valor de K es paramétrico
## Construyo un árbol KD para hacer la búsqueda
kdt = scipy.spatial.cKDTree(comprimidos_total)

## Especifico el número K de vecinos más próximos de cada punto que voy a usar para tener en cuenta
k = 10

## Hago el cálculo de la distancia entre cada punto y sus K vecinos más proximos
dists, neighs = kdt.query(comprimidos_total, k + 1)

## Hago el cálculo del promedio de las distancias de cada punto a su K vecino más próximo
avg_dists = np.mean(dists[:, 1:], axis = 1)

## Hago la graficación de los valores de distancia promedio a los K vecinos más próximos
plt.scatter(np.linspace(0, len(avg_dists) - 1, len(avg_dists)), avg_dists)
plt.show()