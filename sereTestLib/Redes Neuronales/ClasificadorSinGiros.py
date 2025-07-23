## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import keras
import sys
from scipy.signal import *
from sklearn.metrics import confusion_matrix
import scipy.spatial
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from dtwParallel import dtw_functions
from scipy.spatial import distance as d
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
from sklearn.metrics import *
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
from sklearn.preprocessing import *


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

## Especifico la ruta en la cual se encuentran los escalogramas comprimidos que voy a clasificar
ruta_escalogramas = 'C:/Yo/Tesis/sereData/sereData/Dataset/latente_sin_giros'

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Obtengo el conjunto de tramos de marcha sin giros detectados para el paciente
        tramos = os.listdir(ruta_escalogramas + '/S{}'.format(id_persona))

        ## Itero para cada uno de los tramos listados anteriormente
        for i in range (len(tramos)):

            ## Hago la lectura del archivo con el espacio latente
            archivo_comprimido = np.load(ruta_escalogramas + '/S{}/Tramo{}/S{}_latente.npz'.format(id_persona, i, id_persona))

            ## Almaceno el archivo comprimido en una variable como un array bidimensional
            ## La i-ésima fila representa el i-ésimo segmento
            ## La j-ésima columna representa el j-ésimo feature
            espacio_comprimido = archivo_comprimido['X']

            ## En caso de que el segmento detectado tenga más de dos muestras
            if espacio_comprimido.shape[0] > 2:

                ## Recorto una muestra del principio y una muestra del final (para obtener marcha en régimen)
                espacio_comprimido = espacio_comprimido[1: -1, :]

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

## Normalización de la matriz de datos de entrada al algoritmo de clustering
## Hago la normalización por columna es decir por feature
comprimidos_total = normalize(comprimidos_total, norm = "l2", axis = 0)

## ---------------------------------------------- LSTM ------------------------------------------------

# precisiones = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# etiquetas = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# ## Defino un factor de partición
# ## Defino una variable donde guardo el error de prediccion
# error_medio_lstm = 0

# ## Construyo un vector donde guardo la precisión de cada una de las predicciones realizadas
# precisiones = []

# ## Defino una variable donde especifico el numero de subsecuencias
# nro_subsec = 1

# ## Algoritmo de validación cruzada Leave One Out
# ## Itero para cada uno de los pacientes etiquetados para validar el modelo
# for i in range (len(etiquetas)):

#     ## Construyo un modelo secuencial de Keras
#     model = Sequential()

#     ## Agrego una capa de LSTM al modelo especificando la cantidad de hidden units
#     model.add(LSTM(20, activation = 'tanh', input_shape = (None, 256)))

#     ## Agrego una neurona de salida para hacer la clasificacion
#     model.add(Dense(1, activation = 'sigmoid'))

#     ## Compilo el modelo LSTM
#     model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#     ## Itero para cada uno de los pacientes etiquetados para entrenar el modelo
#     for j in range (len(etiquetas)):

#         ## En caso de que el paciente de entrenamiento coincida con el de validación
#         if i == j:

#             ## Sigo con el siguiente paciente de entrenamiento
#             continue

#         ## Hago una partición de la serie del paciente j según el numero dado de subsecuencias
#         comprimidos_particion_j = np.array_split(comprimidos_por_persona[j], nro_subsec)

#         ## Itero para cada uno de los segmentos comprimidos de la persona
#         for segmento in comprimidos_particion_j:

#             ## Hago el ajuste del modelo LSTM para el j-ésimo paciente en su segmento
#             model.fit(np.reshape(normalize(segmento, norm = "l2", axis = 0), (1, segmento.shape[0], segmento.shape[1])),
#                     np.reshape(etiquetas[j], (1, 1)), epochs = 20)
    
#     ## Hago una partición de la serie del paciente i según el numero dado de subsecuencias
#     comprimidos_particion_i = np.array_split(comprimidos_por_persona[i], nro_subsec)

#     ## Itero para cada uno de los segmentos comprimidos de la persona
#     for segmento in comprimidos_particion_i:

#         ## Evalúo la precisión de modelo en el paciente separado para la validación
#         ## Si accuracy = 1 entonces el modelo clasifica correctamente a la muestra
#         ## Si accuracy = 0 entonces el modelo clasifica incorrectamente a la muestra
#         loss, accuracy = model.evaluate(np.reshape(normalize(segmento, norm = "l2", axis = 0), (1, segmento.shape[0], segmento.shape[1])),
#                             np.reshape(etiquetas[i], (1, 1)))

#         ## Me guardo la precisión de la predicción en el vector correspondiente
#         precisiones.append(accuracy)

# ## Construyo una lista que me guarde las predicciones
# predicciones = []

# ## Itero para cada uno de los subsegmentos
# for i in range (len(precisiones)):

#     ## En caso que la predicción haya sido realizada correctamente
#     if precisiones[i] == 1:

#         ## Agrego la etiqueta correcta del segmento correspondiente
#         predicciones.append(etiquetas[i])
    
#     ## En caso de que la predicción no haya sido realizada bien
#     else:

#         ## En caso de que el segmento sea estable verdaderamente
#         if etiquetas[i] == 0:

#             ## Entonces la prediccion realizada es inestable
#             predicciones.append(1)
        
#         ## En caso de que el segmento sea inestable verdaderamente
#         else:

#             ## Entonces la predicción realizada es estable
#             predicciones.append(0)

## ---------------------------------------------- DTW ------------------------------------------------

## Construyo un tensor bidimensional de modo que guarde las distancias relativas
distancias_series = np.zeros((len(comprimidos_por_persona), len(comprimidos_por_persona)))

## Itero para cada uno de las series temporales por persona
for i in range (len(comprimidos_por_persona)):

    ## Itero para las series temporales que no corresponden a la persona (se puede optimizar)
    for j in range (i + 1, len(comprimidos_por_persona)):

        # # Hago la normalización de las columnas del tensor de la persona i
        # serie_i = MeanCenterer().fit(comprimidos_por_persona[i]).transform(comprimidos_por_persona[i])

        # # Hago la normalización de las columnas del tensor de la persona j
        # serie_j = MeanCenterer().fit(comprimidos_por_persona[j]).transform(comprimidos_por_persona[j])

        ## Obtengo la serie asociada al paciente actual
        serie_i = normalize(comprimidos_por_persona[i], norm = "l2", axis = 0)
        # serie_i = comprimidos_por_persona[i]
        # serie_i = StandardScaler().fit_transform(comprimidos_por_persona[i])

        ## Obtengo la serie asociada al paciente a comparar
        serie_j = normalize(comprimidos_por_persona[j], norm = "l2", axis = 0)
        # serie_j = comprimidos_por_persona[j]
        # serie_j = StandardScaler().fit_transform(comprimidos_por_persona[j])

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
            falsos_positivos += 1
        
    ## En caso de que el paciente esté bien clasificado
    else:

        ## En caso de esté etiquetado con 0
        if etiquetas[i] == 0:

            ## Este es un verdadero positivo
            verdaderos_positivos += 1

## Obtengo los verdaderos negativos como los restantes
verdaderos_negativos = len(etiquetas) - (verdaderos_positivos + falsos_negativos + falsos_positivos)

print(verdaderos_negativos)