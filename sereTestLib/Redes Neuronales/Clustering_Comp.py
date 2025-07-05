## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import keras
import sys
from scipy.signal import *
import scipy.spatial
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from dtwParallel import dtw_functions
from scipy.spatial import distance as d
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatosPacientes import *
from scipy.stats import *
from pingouin import multivariate_normality
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.datasets import make_blobs
from mlxtend.preprocessing import MeanCenterer
import susi
import numpy as np
from sklearn_som import som
from pyts.transformation import ShapeletTransform

from pyts.classification import BOSSVS
from pyts.datasets import load_basic_motions
from pyts.multivariate.classification import MultivariateClassifier
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

## ------------------------------------- SELECCIÓN DE MUESTRAS ----------------------------------------------

## Construyo una variable booleana de modo de poder filtrar aquellos pacientes etiquetados
filtrar_etiquetados = True

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

## Genero un vector donde voy a guardar los espacios latentes donde cada elemento es un tensor bidimensional de un paciente
## Entonces me queda un tensor tridimensional (i, j, k) donde:
## i: Hace referencia a la i-ésima persona a la que corresponde el registro
## j: Hace referencia al j-ésimo segmento correspondiente a la persona
## K: Hace referencia a la k-ésima caracteristica (feature)
comprimidos_por_persona = []

## Genero una lista vacía que va a contener las etiquetas de los pacientes
etiquetas = []

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

        ## En caso de que no haya ningun archivo comprimido
        if len(espacio_comprimido) == 0:

            ## Continúo al siguiente paciente
            continue

        ## En caso de que yo quiera filtrar los pacientes etiquetados en mi muestra
        if filtrar_etiquetados:

            ## En caso de que el ID del paciente que está siendo analizado no corresponde a un paciente etiquetado, me lo salteo y no lo proceso
            if id_persona not in id_etiquetados:

                continue

            ## En caso de que el paciente haya sido clasificado como estable
            if id_persona in id_estables:

                ## Concateno un vector de ceros con la cantidad de segmentos que tengo para el registro del paciente
                vector_etiquetas = np.concatenate((np.array(vector_etiquetas), np.zeros((espacio_comprimido.shape[0])))).astype(int)
            
                ## Asigno la etiqueta numérica 0 al paciente estable
                etiquetas.append(0)

            ## En caso de que el paciente haya sido clasificado como inestable
            else:

                ## Concateno un vector de ceros con la cantidad de segmentos que tengo para el registro del paciente
                vector_etiquetas = np.concatenate((np.array(vector_etiquetas), np.ones((espacio_comprimido.shape[0])))).astype(int)

                ## Asigno la etiqueta numérica 1 al paciente inestable
                etiquetas.append(1)

        ## Concateno el espacio comprimido por filas a la matriz donde guardo los espacios latentes totales
        comprimidos_total = np.concatenate((comprimidos_total, espacio_comprimido), axis = 0)

        ## Agrego el espacio comprimido a la lista correspondiente
        comprimidos_por_persona.append(espacio_comprimido)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))

    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue

## Selecciono únicamente aquellas muestras no nulas (elimino el dummy vector)
## Se obtiene entonces una matriz de datos bidimensional donde:
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

# # Grafico la inercia en función de la cantidad de clusters que tengo
# plt.bar(clusters, silhouette_scores)
# plt.xlabel("Numero de Clusters")
# plt.ylabel("Silhouette Score")
# plt.show()

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

# ## Graficación de las distancias de cada punto a los clusters
# plt.scatter(np.linspace(0, len(distancias_puntos) - 1, len(distancias_puntos)), distancias_puntos)
# plt.ylim(0, max(distancias_puntos) * 1.2)
# plt.show()

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

# ## Hago la graficación de los valores de distancia promedio a los K vecinos más próximos
# plt.scatter(np.linspace(0, len(avg_dists) - 1, len(avg_dists)), avg_dists)
# plt.show()

## Test de Hipótesis para la comprobación de normalidad de los vectores comprimidos
test_normal = multivariate_normality(comprimidos_total, alpha = 0.05)

## ---------------------------------------------- LSTM ------------------------------------------------

## Defino un factor de partición
## Defino una variable donde guardo el error de prediccion
error_medio_lstm = 0

## Defino una variable que me de la cantidad de subsecuencias que voy a generar para cada paciente
## En este caso las subsecuencias no se solapan, ya que las transformadas CWT están solapadas entre si
nro_subsec = 1

## Algoritmo de validación cruzada Leave One Out
## Itero para cada uno de los pacientes etiquetados para validar el modelo
for i in range (len(etiquetas)):

    ## Construyo un modelo secuencial de Keras
    model = Sequential()

    ## Agrego una capa de LSTM al modelo especificando la cantidad de hidden units
    model.add(LSTM(20, activation = 'tanh', input_shape = (None, 256)))

    ## Agrego una neurona de salida para hacer la clasificacion
    model.add(Dense(1, activation = 'sigmoid'))

    ## Compilo el modelo LSTM
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    ## Itero para cada uno de los pacientes etiquetados para entrenar el modelo
    for j in range (len(etiquetas)):

        ## En caso de que el paciente de entrenamiento coincida con el de validación
        if i == j:

            ## Sigo con el siguiente paciente de entrenamiento
            continue

        ## Hago una partición de la serie del paciente j según el numero dado
        comprimidos_particion_j = np.array_split(comprimidos_por_persona[j], nro_subsec)

        ## Itero para cada uno de los segmentos comprimidos de la persona
        for segmento in comprimidos_particion_j:

            ## Hago el ajuste del modelo LSTM para el j-ésimo paciente en su segmento
            model.fit(np.reshape(segmento, (1, segmento.shape[0], segmento.shape[1])),
                    np.reshape(etiquetas[j], (1, 1)), epochs = 20)
    
    ## Hago una partición de la serie del paciente j según el numero dado
    comprimidos_particion_i = np.array_split(comprimidos_por_persona[i], nro_subsec)

    ## Itero para cada uno de los segmentos comprimidos de la persona
    for segmento in comprimidos_particion_i:

        ## Evalúo la precisión de modelo en el paciente separado para la validación
        loss, accuracy = model.evaluate(np.reshape(segmento, (1, segmento.shape[0], segmento.shape[1])),
                            np.reshape(etiquetas[i], (1, 1)))

## ---------------------------------------------- DTW ------------------------------------------------

## Construyo un tensor bidimensional de modo que guarde las distancias relativas
distancias_series = np.zeros((len(comprimidos_por_persona), len(comprimidos_por_persona)))

## Itero para cada uno de las series temporales por persona
for i in range (len(comprimidos_por_persona)):

    ## Itero para las series temporales que no corresponden a la persona (se puede optimizar)
    for j in range (i + 1, len(comprimidos_por_persona)):

        ## Hago la normalización de las columnas del tensor de la persona i
        ## serie_i = MeanCenterer().fit(comprimidos_por_persona[i]).transform(comprimidos_por_persona[i])

        ## Hago la normalización de las columnas del tensor de la persona j
        ## serie_j = MeanCenterer().fit(comprimidos_por_persona[j]).transform(comprimidos_por_persona[j])

        ## Obtengo la serie asociada al paciente actual (sin normalizar)
        serie_i = comprimidos_por_persona[i]

        ## Obtengo la serie asociada al paciente a comparar (sin normalizar)
        serie_j = comprimidos_por_persona[j]

        ## CÁLCULO DE LA DTW entre ambas series
        ## Recuerdo que las filas de la matriz son las observaciones
        ## Recuerdo que las columnas de la matriz son las variables
        ## Para que la DTW esté bien definida los tensores deben tener mismo numero de variables (o sea de columnas)
        ## El DTW se encuentra bien definido incluso cuando los tensores tienen distinto número de filas
        dist = dtw_functions.dtw(serie_i, serie_j, type_dtw = "d", local_dissimilarity = d.euclidean, MTS = True)

        ## Asigno la distancia a la posición de la matriz correspondiente
        distancias_series[i, j] = dist
        
        ## Como la matriz de distancias es simétrica hago lo mismo con la otra posicion
        distancias_series[j, i] = dist

## ---------------------------------------- VALIDACIÓN ------------------------------------------------

## POSITIVO = PACIENTE INESTABLE
## NEGATIVO = PACIENTE ESTABLE
## Variable que contabiliza los Falsos Positivos
falsos_positivos = 0

## Variable que contabiliza los Falsos Negativos
falsos_negativos = 0

## Variable que contabiliza los Verdaderos Positivos
verdaderos_positivos = 0

## Itero para cada uno de las series temporales por persona
for i in range (len(comprimidos_por_persona)):

    ## Obtengo la posición del argumento minimo no nulo en donde se presenta la distancia deseada
    posicion_minimo = np.argmin(distancias_series[i, :][distancias_series[i, :] != 0])

    ## En caso de que el paciente esté mal clasificado
    if etiquetas[posicion_minimo + 1] != etiquetas[i]:

        ## Si el paciente estaba etiquetado con 1 (positivo)
        if etiquetas[i] == 1:

            ## Este es un falso negativo
            falsos_negativos += 1
        
        ## Si el paciente estaba etiquetado con 0 (negativo)
        elif etiquetas[i] == 0:

            ## Este es un falso positivo
            falsos_negativos += 1
        
    ## En caso de que el paciente esté bien clasificado
    else:

        ## En caso de esté etiquetado con 1
        if etiquetas[i] == 0:

            ## Este es un verdadero positivo
            verdaderos_positivos += 1

## Obtengo los verdaderos negativos como los restantes
verdaderos_negativos = len(etiquetas) - (verdaderos_positivos + falsos_negativos + falsos_positivos)

## --------------------------------------------- SOM ------------------------------------------------

som = susi.SOMClustering()
som.fit(comprimidos_total)

u_matrix = som.get_u_matrix()
plt.imshow(np.squeeze(u_matrix), cmap="Greys")
plt.colorbar()
plt.show()