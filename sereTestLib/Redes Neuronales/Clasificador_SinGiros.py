## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
from parameters import *
import keras
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
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, KFold

## Defino una función que me permita ver si dos matrices tienen al menos una fila en común
def FilasEnComun(matriz1, matriz2):

    ## Itero para todas las filas de matriz1
    for i in range (matriz1.shape[0]):

        ## Itero para todas las filas de matriz2
        for j in range (matriz2.shape[0]):

            ## En caso de que exista una fila en común
            if (matriz1[i, :] == matriz2[j, :]).all():

                ## Retorno True
                return True
    
    ## En caso de que no haya filas en común, retorno False
    return False

## Construyo una función que me compruebe que dos secuencias de marcha pertenecen a la misma persona
def SecuenciasMismaPersona(sec1, sec2, secuencias_por_persona):

    ## Itero para cada una de las personas
    for sec_persona in secuencias_por_persona:

        ## En caso de que ambas secuencias pertenezcan a la misma persona
        if any(np.array_equal(sec1, i) for i in sec_persona) and any(np.array_equal(sec2, i) for i in sec_persona):

            ## Entonces ambas secuencias pertenecen a la misma persona
            return True
    
    ## En caso de que las secuencias pertenezcan a personas distintas, devuelvo False
    return False

## Creo una función que me haga el cargado de datos comprimidos
def CargadoComprimidos(filtrar_etiquetados):

    ## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## Creo un vector donde voy a guardar las etiquetas asociadas a cada uno de los pacientes etiquetados, respetando el orden
    vector_etiquetas = []

    ## Genero una lista vacía que va a contener las etiquetas de los pacientes
    etiquetas = []

    ## Genero un vector vacío para poder concatenar los vectores con todas las representaciones latentes
    comprimidos_total = np.zeros((1, 256))

    ## Genero una lista una lista donde el elemento i es una matriz (j, k) donde:
    ## i: Hace referencia a la i-ésima secuencia de marcha
    ## j: Hace referencia a la j-ésima observación temporal de dicha secuencia de marcha
    ## K: Hace referencia a la k-ésima feature
    secuencias_individuales = []

    ## Genero una lista una lista donde el elemento i es un tensor (j, k, l) donde:
    ## i: Hace referencia a la i-ésima persona
    ## j: Hace referencia a la j-ésima secuencia de la persona
    ## k: Hace referencia a la k-ésima observación temporal de dicha secuencia
    ## K: Hace referencia a la k-ésima feature
    secuencias_por_persona = [[]]

    ## Itero para cada uno de los identificadores de los pacientes
    for id_persona in ids_existentes:

        ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
        try:

            ## Obtengo el conjunto de tramos de marcha sin giros detectados para el paciente
            tramos = os.listdir(ruta_comprimidas_sin_giros + '/S{}'.format(id_persona))

            ## Itero para cada uno de los tramos listados anteriormente
            for i in range (len(tramos)):

                ## Hago la lectura del archivo con el espacio latente
                archivo_comprimido = np.load(ruta_comprimidas_sin_giros + '/S{}/Tramo{}/S{}_latente.npz'.format(id_persona, i, id_persona))

                ## Almaceno el archivo comprimido en una variable como un array bidimensional
                ## La i-ésima fila representa el i-ésimo segmento
                ## La j-ésima columna representa el j-ésimo feature
                espacio_comprimido = archivo_comprimido['X']

                ## En caso de que el segmento detectado tenga la cantidad necesaria de muestras
                if espacio_comprimido.shape[0] > 2:

                    ## Recorto muestras al principio y al final
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

                    ## Agrego la secuencia individual a la lista de secuencias individuales
                    secuencias_individuales.append(espacio_comprimido)

                    ## Agrego el espacio comprimido a la lista correspondiente
                    secuencias_por_persona[-1].append(espacio_comprimido)

            ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
            print("ID del paciente que está siendo procesado: {}".format(id_persona))

            ## Agrego una lista al final de las secuencias por persona
            secuencias_por_persona.append([])

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

    ## Elimino el úlimo elemento de la lista de secuencias por personas, que es una lista vacía
    secuencias_por_persona = secuencias_por_persona[:-1]

    ## Retorno el vector de secuencias individuales, las etiquetas asociadas, las secuencias por persona y los comprimidos totales
    return etiquetas, secuencias_individuales, secuencias_por_persona, comprimidos_total

## Creo una función que me haga el entrenamiento y validación de la red LSTM con LOO por paciente
def ValidacionLSTM(etiquetas, secuencias_individuales):

    ## Construyo un vector donde guardo la precisión de cada una de las predicciones realizadas
    ## El i-ésimo elemento de dicho vector va a ser la precisión en el i-ésimo fold
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
            comprimidos_particion_j = np.array_split(secuencias_individuales[j], nro_subsec)

            ## Itero para cada uno de los segmentos comprimidos de la persona
            for segmento in comprimidos_particion_j:

                ## Hago el ajuste del modelo LSTM para el j-ésimo paciente en su segmento
                model.fit(np.reshape(normalize(segmento, norm = "l2", axis = 0), (1, segmento.shape[0], segmento.shape[1])),
                        np.reshape(etiquetas[j], (1, 1)), epochs = 20)
        
        ## Hago una partición de la serie del paciente i según el numero dado de subsecuencias
        comprimidos_particion_i = np.array_split(secuencias_individuales[i], nro_subsec)

        ## Itero para cada uno de los segmentos comprimidos de la persona
        for segmento in comprimidos_particion_i:

            ## Evalúo la precisión de modelo en el paciente separado para la validación
            ## Si accuracy = 1 entonces el modelo clasifica correctamente a la muestra
            ## Si accuracy = 0 entonces el modelo clasifica incorrectamente a la muestra
            loss, accuracy = model.evaluate(np.reshape(normalize(segmento, norm = "l2", axis = 0), (1, segmento.shape[0], segmento.shape[1])),
                                np.reshape(etiquetas[i], (1, 1)))

            ## Me guardo la precisión de la predicción en el vector correspondiente
            precisiones.append(accuracy)

    ## Construyo una lista que me guarde las predicciones
    predicciones_lstm = []

    ## Itero para cada uno de los subsegmentos
    for i in range (len(precisiones)):

        ## En caso que la predicción haya sido realizada correctamente
        if precisiones[i] == 1:

            ## Agrego la etiqueta correcta del segmento correspondiente
            predicciones_lstm.append(etiquetas[i])

        ## En caso de que la predicción no haya sido realizada bien
        else:

            ## En caso de que el segmento sea estable verdaderamente
            if etiquetas[i] == 0:

                ## Entonces la prediccion realizada es inestable
                predicciones_lstm.append(1)

            ## En caso de que el segmento sea inestable verdaderamente
            else:

                ## Entonces la predicción realizada es estable
                predicciones_lstm.append(0)
    
    ## Retorno las predicciones realizadas haciendo la validación LSTM
    return predicciones_lstm

## Creo una función que me haga la validación con DTW 1NN
def ValidacionDTW(etiquetas, secuencias_individuales, secuencias_por_persona):

    ## Construyo un tensor bidimensional de modo que guarde las distancias relativas
    distancias_series = np.zeros((len(secuencias_individuales), len(secuencias_individuales)))

    ## Itero para cada uno de las series temporales por persona
    for i in range (len(secuencias_individuales)):

        ## Itero para las series temporales que no corresponden a la persona (se puede optimizar)
        for j in range (i + 1, len(secuencias_individuales)):

            ## Obtengo la serie asociada al paciente actual y la normalizo por feature
            serie_i = normalize(secuencias_individuales[i], norm = "l2", axis = 0)

            ## Obtengo la serie asociada al paciente a comparar y la normalizo por feature
            serie_j = normalize(secuencias_individuales[j], norm = "l2", axis = 0)

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

    ## Genero un vector donde me guardo las predicciones
    predicciones_dtw = []

    ## Genero una lista donde el i-ésimo valor será True si la secuencia más próxima a la secuencia i
    ## pertenece a la misma persona. Lo veo como un indicador de variabilidad intra-persona
    secuencias_misma_persona = []

    ## Itero para cada uno de las series temporales por persona
    for i in range (len(secuencias_individuales)):

        ## Obtengo la posición del argumento minimo no nulo en donde se presenta la distancia deseada
        posicion_minimo = np.argmin(distancias_series[i, :][distancias_series[i, :] != 0])

        ## Agrego la prediccion como la etiqueta del más cercano
        predicciones_dtw.append(etiquetas[posicion_minimo + 1])

        ## Obtengo la secuencia de distancia mínima a la secuencia actual
        secuencia_mas_proxima = secuencias_individuales[posicion_minimo + 1]

        ## Miro a ver si la secuencia de distancia mínima pertenece a la misma persona
        misma_persona = SecuenciasMismaPersona(secuencias_individuales[i], secuencia_mas_proxima, secuencias_por_persona)

        ## Me guardo el valor booleano si la secuencia es la misma persona, en la lista dada
        secuencias_misma_persona.append(misma_persona)

    ## Hago la conversión a vector numpy de los valores booleanos de las secuencias pertenecientes a la misma persona
    secuencias_misma_persona = np.array(secuencias_misma_persona)

    ## Convierto el vector de predicciones en vector numpy
    predicciones_dtw = np.array(predicciones_dtw)

    ## Retorno las predicciones realizadas por la validación DTW
    return predicciones_dtw

## Creo una función la cual me construya automáticamente la matriz de confusión dadas las etiqetas y predicciones
def MatrizConfusion(etiquetas, predicciones):

    ## Instrucciones de construcción y display de la matriz de confusion
    cm = confusion_matrix(np.array(etiquetas), predicciones)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Estable', 'Inestable'])
    disp.plot()
    plt.show()

## Ejecución principal del programa
if __name__== '__main__':

    ## Hago el cargado de datos comprimidos
    etiquetas, secuencias_individuales, secuencias_por_persona = CargadoComprimidos(filtrar_etiquetados = True)

    ## Hago la validación del algoritmo de clasificacion DTW 1NN
    predicciones = ValidacionDTW(etiquetas, secuencias_individuales, secuencias_por_persona)

    ## Construyo la matriz de confusión correspondiente
    MatrizConfusion(etiquetas, predicciones)