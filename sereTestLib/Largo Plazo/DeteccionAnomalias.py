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
from scipy.stats import *
from pyod.models.knn import KNN
import numpy
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance, MinCovDet

## -------------------------------- DISCRIMINACIÓN REPOSO - ACTIVIDAD --------------------------------------

## La idea de ésta parte consiste en poder hacer una discriminación entre reposo y actividad
## Especifico la ruta en la cual se encuentra el registro a leer
ruta_registro = 'C:/Yo/Tesis/sereData/sereData/Registros/Actividades_Rodrigo.txt'

##  Hago la lectura de los datos
data, acel, gyro, cant_muestras, periodoMuestreo, tiempo = LecturaDatos(id_persona = 299, lectura_datos_propios = True, ruta = ruta_registro)

## Defino la cantidad de muestras de la ventana que voy a tomar
muestras_ventana = 200

## Defino la cantidad de muestras de solapamiento entre ventanas
muestras_solapamiento = 100

## Hago el cálculo del vector de SMA para dicha persona
vector_SMA, features, ventanas = DeteccionActividades(acel, tiempo, muestras_ventana, muestras_solapamiento, periodoMuestreo, cant_muestras, actividad = None)

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
        tramos_actividades.append([tramos_actividades[-1][1], i + 1])

## Agrego el tramo faltante del procedimiento anterior
tramos_actividades.append([tramos_actividades[-1][1], i + 1])

## Elimino el dummy vector del inicio y lo transformo en un vector numpy
## Obtengo una matriz donde:
## La i-ésima fila hace referencia al i-ésimo segmento uniforme
## La columna 0 es el indice de la posición inicial, la columna 1 es el indice de posicion final
tramos_actividades = np.array((tramos_actividades[1:]))

## ------------------------------------------ GRAFICACIÓN ---------------------------------------------

## La idea es poder graficar las señales de acelerómetros discriminando entre movimiento y reposo
## Itero para cada uno de los segmentos tomados
for i in range (tramos_actividades.shape[0]):

    ## En caso de que dicho tramo corresponda a movimiento
    if pat_predictions[tramos_actividades[i, 0]] == 1:
        
        ## Hago la graficación en rojo del tramo de movimiento detectado en los tres acelerómeteros
        plt.plot(tiempo[ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]],
                acel[:, 0][ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]], color = 'r', label = 'Movimiento')
        plt.plot(tiempo[ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]],
                acel[:, 1][ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]], color = 'r')
        plt.plot(tiempo[ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]],
                acel[:, 2][ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]], color = 'r')

    ## En caso de que dicho tramo corresponda a reposo
    else:

        ## Hago la graficación en rojo del tramo de movimiento detectado en los tres acelerómeteros
        plt.plot(tiempo[ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]],
                acel[:, 0][ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]], color = 'b', label = 'Reposo')
        plt.plot(tiempo[ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]],
                acel[:, 1][ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]], color = 'b')
        plt.plot(tiempo[ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]],
                acel[:, 2][ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][0, 0] : ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1]][-1, -1]], color = 'b')

## Despliego la gráfica y configuro los parámetros de Leyendas
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.xlabel('Tiempo(s)')
plt.ylabel("Aceleracion $(m/s^2)$")
plt.legend(by_label.values(), by_label.keys())
plt.title('Discriminación Movimiento-Reposo')
plt.show()

## ---------------------------------------- PREPROCESAMIENTO -------------------------------------------

## Hago la conversión de los features a un array bidimensional
## Obtengo una matriz donde:
## La i-ésima fila hace referencia al i-ésimo segmento
## La j-ésima columna hace referencia al j-ésimo feature
features = np.reshape(features, (np.array(features).shape[0], np.array(features).shape[2]))

## Hago la normalización de la matriz de features
features_norm = normalize(features, norm = "l2", axis = 0)

## Especifico una lista con la cantidad de clusters que voy a usar durante el análisis
clusters = np.linspace(2, 10, 9).astype(int)

## Genero un objeto el cual realizará el PCA
pca_init = PCA()

## Ajusto el modelo de PCA a los datos
pca_init.fit(features_norm)

## Genero una variable donde guardo la suma de la explicación de la varianza del dataset por las features
suma_exp_varianza = pca_init.explained_variance_ratio_.cumsum()

## Grafico la varianza explicada por los componentes
plt.figure(figsize = (10, 8))
plt.plot(range(1, features_norm.shape[1] + 1), suma_exp_varianza, marker = 'o', linestyle = '--')
plt.title('Varianza Explicada por los componentes')
plt.xlabel('Número de componentes')
plt.ylabel('Varianca Explicada Acumulativa')
plt.show()

## Construyo una variable que me va a decir la cantidad de componentes que tengo
componentes = 0

## Itero para cada una de las proporciones explicativas de la varianza
for i in range (len(suma_exp_varianza)):
    
    ## Aumento en una unidad la cantidad de componentes
    componentes += 1

    ## En caso de que la suma total de la varianza explicada sea mayor a 0.8
    if suma_exp_varianza[i] > 0.8:

        ## Termino el bucle
        break

## Ahora aplico PCA con la cantidad de componentes que seleccioné con el criterio anterior
pca = PCA(n_components = componentes)

## Ajusto el modelo a mis datos con la cantidad seleccionada de componentes
pca.fit(features_norm)

## Hago la selección de los features transformando según las componentes principales halladas
features_norm = pca.transform(features_norm)

## ------------------------------------------- ANOMALÍAS ----------------------------------------------

## Construyo una matriz donde voy a guardar las anomalías detectadas usando el método de clustering
anomalias = np.zeros((1, 2))

## Construyo una matriz donde voy a guardar las anomalias detectadas usando knn
anomalias_nn = np.zeros((1, 2))

## Itero para cada una de los tramos que tengo detectados
for i in range (tramos_actividades.shape[0]):

    ## En caso de que el segmento tenga menos de una cantidad predeterminada de ventanas
    if features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]].shape[0] <= 10:

        ## Concateno la posicion de las anomalias detectadas en el tramo con las previas
        anomalias = np.concatenate((anomalias, ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1],:]))

        ## Concateno la posicion de las anomalias detectadas en el tramo con las previas para NN
        anomalias_nn = np.concatenate((anomalias_nn, ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1],:]))

        ## Sigo con el siguiente segmento
        continue

    ## Construyo un vector en donde voy a guardar los silhouette scores para cada una de las distribuciones de clusters
    silhouette_scores = []

    ## Itero para cada una de las cantidades de clusters que tengo
    for nro_clusters in clusters:

        ## En caso de que el número de clusters en la iteración sea mayor a la cantidad de puntos en el dataset
        if nro_clusters > features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]].shape[0] - 1:

            ## Concateno la posicion de las anomalias detectadas en el tramo con las previas
            anomalias = np.concatenate((anomalias, ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1],:]))

            ## Concateno la posicion de las anomalias detectadas en el tramo con las previas para NN
            anomalias_nn = np.concatenate((anomalias_nn, ventanas[tramos_actividades[i, 0] : tramos_actividades[i, 1],:]))

            continue

        ## Aplico un clustering KMeans a los datos correspondientes de entrada
        kmeans = KMeans(n_clusters = nro_clusters, random_state = 0, n_init = "auto").fit(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

        ## Agrego la silhouette score a la lista de indicadores
        silhouette_scores.append(silhouette_score(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]], kmeans.fit_predict(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])))

    ## Obtengo la cantidad óptima de clústers observando aquel número en donde se maximice la silhouette score
    clusters_optimo = np.argmax(silhouette_scores) + 2

    ## Aplico el clústering KMeans pasando como entrada el número óptimo de clústers determinado por el Silhouette Score
    kmeans = KMeans(n_clusters = clusters_optimo, random_state = 0, n_init = "auto").fit(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

    ## Construyo un vector donde voy a guardar la distancia euclideana de cada punto a su respectivo centroide
    distancias_puntos = []

    ## Construyo un vector en donde voy a guardar las matrices de covarianza de los clusters
    matrices_cov = []

    ## Itero para cada uno de los clusters que tengo
    for cluster in range (clusters_optimo):

        ## Obtengo todos los puntos correspondientes al cluster
        puntos_cluster = features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]][np.where(kmeans.labels_ == cluster)]

        ## En caso de que la cantidad de puntos en el cluster sea mayor a 1
        if puntos_cluster.shape[0] > 1:

            ## Hago el cálculo de la matriz de covarianza del clúster
            matriz_cov = MinCovDet().fit(puntos_cluster)

            ## Agrego la matriz de covarianza a la lista correspondiente
            matrices_cov.append(matriz_cov.covariance_)
        
        ## En otro caso
        else:

            ## Asigno como matriz de covarianza la matriz identidad
            matriz_cov = np.identity(puntos_cluster.shape[1])

            ## La agrego a la lista de matrices
            matrices_cov.append(matriz_cov)

    ## Itero para cada uno de los puntos del dataset
    for j in range (len(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])):

        ## Obtengo el centroide del clúster que está asociado al i-ésimo punto
        centroide = kmeans.cluster_centers_[kmeans.labels_[j]]

        ## Calculo la distancia de Mahalanobis al cuadrado entre el i-ésimo punto y el centroide asociado
        distancia = np.dot((features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]][j] - centroide), 
                        np.matmul(np.linalg.inv(matrices_cov[0]), (features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]][j] - centroide)))

        ## Agrego la distancia del punto a su centroide a la lista
        distancias_puntos.append(distancia)

    ## Hago la transformación del vector a numpy array
    distancias_puntos = np.array(distancias_puntos)

    ## Defino el umbral de distancia a partir del cual yo considero que van a ser anomalías
    umbral = np.percentile(distancias_puntos, 95)
    # umbral = np.mean(distancias_puntos) + 2 * np.std(distancias_puntos)

    ## Obtengo el valor de las distancias a los centroides de los puntos que considero anomalías
    anomalias_tramo = distancias_puntos[distancias_puntos > umbral]

    ## Concateno la posicion de las anomalias detectadas en el tramo con las previas
    anomalias = np.concatenate((anomalias, ventanas[np.where(np.isin(distancias_puntos, anomalias_tramo))[0] + tramos_actividades[i, 0]]))

    ## Genero el modelo KNN para mis datos
    knn_model = KNN()
    
    ## Hago la predicción según el modelo de KNN
    knn_df = knn_model.fit_predict(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

    ## Obtengo las filas de las ventanas en las cuales se detectaron las anomalias
    anomalias_knn = np.where(np.isin(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]], 
                                    features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]][knn_df == 1])[:,0] == True)
    
    ## Obtengo las posiciones absolutas de las anomalias en el array de ventanas
    anomalias_knn = ventanas[anomalias_knn + tramos_actividades[i, 0],:]

    ## Concateno la posicion de las anomalias detectadas en el tramo con las previas para NN
    anomalias_nn = np.concatenate((anomalias_nn, np.reshape(anomalias_knn, (anomalias_knn.shape[1], anomalias_knn.shape[2]))))

    ## Diagrama de dispersión de las distancias a los centroides
    plt.figure(figsize = (10, 6))
    plt.scatter(np.linspace(0, len(distancias_puntos) - 1, len(distancias_puntos)), distancias_puntos, c = kmeans.labels_, cmap='viridis')
    plt.scatter(np.where(np.isin(distancias_puntos, anomalias_tramo))[0], anomalias_tramo, c = 'red')
    plt.show()
    
    ## Especifico la cantidad de vecinos
    neighbors = NearestNeighbors(n_neighbors = 4)

    ## Hago el ajuste de los vecinos más cercanos de mis datos
    neighbors_fit = neighbors.fit(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

    ## Calculo las distancias a los k vecinos más próximos
    distances, indices = neighbors_fit.kneighbors(features_norm[tramos_actividades[i, 0] : tramos_actividades[i, 1]])

#     # Graficación de distancia al k-vecino más proximo en función de la cantidad de puntos
#     distances = np.sort(distances[:, 3], axis = 0)
#     plt.plot(distances)
#     plt.ylabel('k-distancia')
#     plt.xlabel('Puntos ordenados por distancia')
#     plt.show()

#     plt.plot(np.diff(distances))
#     plt.ylabel('k-distancia')
#     plt.xlabel('Puntos ordenados por distancia')
#     plt.show()

## Elimino el dummy vector inicial de la matriz de anomalias para clustering
anomalias = anomalias[1:,:]

## Elimino el dummy vector inicial de la matriz de anomalias para NN
anomalias_nn = anomalias_nn[1:,:]

## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
plt.plot(tiempo, acel[:,0], color = 'r', label = '$a_x$')
plt.plot(tiempo, acel[:,1], color = 'b', label = '$a_y$')
plt.plot(tiempo, acel[:,2], color = 'g', label = '$a_z$')

## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleracion $(m/s^2)$")

## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
plt.legend()

## Muestro la gráfica
plt.show()

## ------------------------------------------ GRAFICACIÓN ---------------------------------------------

## La idea es poder discriminar entre segmentos anómalos y no anómalos
## Itero para cada una de las ventanas
for i in range (ventanas.shape[0]):

    ## En caso de que la ventana haya sido categorizada como anómala
    if ventanas[i, :] in anomalias:
        
        ## Hago la graficación en rojo de la ventana anómala detectada en los tres acelerómetros
        plt.plot(tiempo[ventanas[i, :][0] : ventanas[i, :][1]],
                acel[:, 0][ventanas[i, :][0] : ventanas[i, :][1]], color = 'r', label = 'Anomalía')
        plt.plot(tiempo[ventanas[i, :][0] : ventanas[i, :][1]],
                acel[:, 1][ventanas[i, :][0] : ventanas[i, :][1]], color = 'r')
        plt.plot(tiempo[ventanas[i, :][0] : ventanas[i, :][1]],
                acel[:, 2][ventanas[i, :][0] : ventanas[i, :][1]], color = 'r')

    ## En caso de que dicha ventana no haya sido clasificada como anómala
    else:

        ## Hago la graficación en rojo del tramo de movimiento detectado en los tres acelerómeteros
        plt.plot(tiempo[ventanas[i, :][0] : ventanas[i, :][1]],
                acel[:, 0][ventanas[i, :][0] : ventanas[i, :][1]], color = 'b', label = 'Normal')
        plt.plot(tiempo[ventanas[i, :][0] : ventanas[i, :][1]],
                acel[:, 1][ventanas[i, :][0] : ventanas[i, :][1]], color = 'b')
        plt.plot(tiempo[ventanas[i, :][0] : ventanas[i, :][1]],
                acel[:, 2][ventanas[i, :][0] : ventanas[i, :][1]], color = 'b')

## Despliego la gráfica
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.xlabel('Tiempo(s)')
plt.ylabel("Aceleracion $(m/s^2)$")
plt.legend(by_label.values(), by_label.keys())
plt.title('Deteccion Anomalias')
plt.show()