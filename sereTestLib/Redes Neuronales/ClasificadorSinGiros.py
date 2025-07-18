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

## Defino un factor de partición
## Defino una variable donde guardo el error de prediccion
error_medio_lstm = 0

## Construyo un vector donde guardo la precisión de cada una de las predicciones realizadas
precisiones = []

## Defino una variable donde especifico el numero de subsecuencias
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

        ## Hago una partición de la serie del paciente j según el numero dado de subsecuencias
        comprimidos_particion_j = np.array_split(comprimidos_por_persona[j], nro_subsec)

        ## Itero para cada uno de los segmentos comprimidos de la persona
        for segmento in comprimidos_particion_j:

            ## Hago el ajuste del modelo LSTM para el j-ésimo paciente en su segmento
            model.fit(np.reshape(segmento, (1, segmento.shape[0], segmento.shape[1])),
                    np.reshape(etiquetas[j], (1, 1)), epochs = 100)
    
    ## Hago una partición de la serie del paciente i según el numero dado de subsecuencias
    comprimidos_particion_i = np.array_split(comprimidos_por_persona[i], nro_subsec)

    ## Itero para cada uno de los segmentos comprimidos de la persona
    for segmento in comprimidos_particion_i:

        ## Evalúo la precisión de modelo en el paciente separado para la validación
        ## Si accuracy = 1 entonces el modelo clasifica correctamente a la muestra
        ## Si accuracy = 0 entonces el modelo clasifica incorrectamente a la muestra
        loss, accuracy = model.evaluate(np.reshape(segmento, (1, segmento.shape[0], segmento.shape[1])),
                            np.reshape(etiquetas[i], (1, 1)))
    
        ## Me guardo la precisión de la predicción en el vector correspondiente
        precisiones.append(accuracy)