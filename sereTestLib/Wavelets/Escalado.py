## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
import os
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from Escalogramas import *

## --------------------------------------------- ESCALADO ----------------------------------------------

## Especifico la ruta de salida de los archivos
ruta_guardado =  'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train' + directorio_muestra

## Especifico la ruta de salida de escalogramas individuales
ruta_guardado_individual =  'C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_ind_nuevo/train' + directorio_muestra

## En caso de que el directorio no exista
if not os.path.exists(ruta_guardado):

    ## Creo el directorio correspondiente
    os.makedirs(ruta_guardado)

## En caso de que el directorio no exista
if not os.path.exists(ruta_guardado_individual):

    ## Creo el directorio correspondiente
    os.makedirs(ruta_guardado_individual)

## Especifico la cantidad de segmentos que tengo
segments = np.shape(escalogramas_segmentos)[0]

## <<global_max_acc>> me va a dar el máximo global de las señales de acelerómetros en cada uno de los tres ejes
## Inicialmente este vector variable se inicializa todo en cero para que luego se pueda sobreescribir
global_max_acc = [0, 0, 0]

## <<global_max_gyro>> me va a dar el máximo global de las señales de giroscopios en cada uno de los tres ejes
## Inicialmente este vector variable se inicializa todo en cero para que luego se pueda sobreescribir
global_max_gyro = [0, 0, 0]

## <<global_min_acc>> me va a dar el minimo global de las señales de acelerometros en cada uno de los tres ejes
## Inicialmente este vector variable se inicializa todo en mil para que luego se pueda sobreescribir
global_min_acc = [1000, 1000, 1000]

## <<global_min_gyro>> me va a dar el minimo global de las señales de acelerometros en cada uno de los tres ejes
## Inicialmente este vector variable se inicializa todo en mil para que luego se pueda sobreescribir
global_min_gyro = [1000, 1000, 1000]

## Itero para cada valor de segmento en el rango 0, 1, ..., segments - 1
for segment in np.arange(segments):

    ## <<escalogramas>> es la lista de diccionarios cuyo i-esimo elemento es el diccionario asociado al i-ésimo segmento
    ## <<segment>> me da el identificador del segmento
    ## X va a ser el tensor tridimensional correspondiente al i-ésimo segmento (dimensiones frecuencia-tiempo-canales)
    X = escalogramas_segmentos[segment]['escalograma']

    ## Hago el cálculo de los máximos y mínimos LOCALES AL ESCALOGRAMA ACTUAL
    ## Recuerdo que en cada escalograma tridimensional las tres capas superiores contienen los coeficientes de las aceleraciones
    ## Por otro lado las tres capas inferiores contienen los coeficientes de los giroscopios
    ## <<local_max_acc>> va a contener el elemento de mayor valor absoluto presente en cada matriz de coeficientes de wavelet de las aceleraciones
    ## Esto implica que <<local_max_acc>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada maximo
    local_max_acc =  [np.max(np.abs(X[0, :, :])), np.max(np.abs(X[1, :, :])), np.max(np.abs(X[2, :, :]))]

    ## <<local_max_gyro>> va a contener el elemento de mayor valor absoluto presente en cada matriz de coeficientes de wavelet de los giroscopios
    ## Esto implica que <<local_max_gyro>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada maximo
    local_max_gyro = [np.max(np.abs(X[3, :, :])), np.max(np.abs(X[4, :, :])), np.max(np.abs(X[5, :, :]))]

    ## <<local_min_acc>> va a contener el elemento de menor valor absoluto presente en cada matriz de coeficientes de wavelet de las aceleraciones
    ## Esto implica que <<local_min_acc>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada minimo
    local_min_acc =  [np.min(np.abs(X[0, :, :])), np.min(np.abs(X[1, :, :])), np.min(np.abs(X[2, :, :]))]            

    ## <<local_min_gyro>> va a contener el elemento de menor valor absoluto presente en cada matriz de coeficientes de wavelet de los giroscopios
    ## Esto implica que <<local_min_gyro>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada minimo
    local_min_gyro = [np.min(np.abs(X[3, :, :])), np.min(np.abs(X[4, :, :])), np.min(np.abs(X[5, :, :]))]         

    ## Itero para cada una de las 3 coordenadas de los vectores anteriores
    ## Tengo 3 coordenadas porque tengo 3 canales de aceleración y 3 canales de giroscopios
    ## El propósito de ésto es actualizar los máximos y minimos valores de coeficientes globales en caso de que se vayan actualizando
    ## Con "coeficientes globales" me refiero a los coeficientes de TODOS los escalogramas de un mismo paciente
    ## Con "coeficientes locales" me refiero a los coeficientes del escalograma actual en la iteracion
    for i in range(3):

        ## En caso de que la i-ésima aceleración maxima del escalograma actual supere al valor global, lo actualizo
        if local_max_acc[i] > global_max_acc[i]:

            ## Actualización del máximo correspondiente
            global_max_acc[i] = local_max_acc[i]
        
        ## En caso de que el i-esimo valor de giroscopio maximo del escalograma actual supere al valor global, lo actualizo
        if local_max_gyro[i] > global_max_gyro[i]:

            ## Actualización del máximo correspondiente
            global_max_gyro[i] = local_max_gyro[i]
        
        ## En caso de que la i-ésima aceleración minima del escalograma actual sea inferior al valor global, lo actualizo
        if local_min_acc[i] < global_min_acc[i]:

            ## Actualización del mínimo correspondiente
            global_min_acc[i] = local_min_acc[i]
        
        ## En caso de que el i-esimo valor de giroscopio minimo del escalograma actual sea inferior al valor global, lo actualizo
        if local_min_gyro[i] < global_min_gyro[i]:

            ## Actualización del mínimo correspondiente
            global_min_gyro[i] = local_min_gyro[i]

    ## Itero para cada uno de los canales (tengo 6 canales que son tres aceleraciones y tres giroscopios)
    for i in range(X.shape[0]):

        ## En caso de que me esté refiriendo a las aceleraciones
        if i < 3:

            ## TRANSFORMACIÓN DE LOS ESCALOGRAMAS A INTENSIDADES DE GRIS
            X[i, :, :] = np.rint((X[i, :, :] - global_min_acc[i]) * 256 / (global_max_acc[i] - global_min_acc[i])).astype(np.intc)

        ## En caso de que me esté refiriendo a los giroscopios
        else:

            ## TRANSFORMACIÓN DE LOS ESCALOGRAMAS A INTENSIDADES DE GRIS
            X[i, :, :] = np.rint((X[i, :, :] - global_min_gyro[i - 3]) * 256 / (global_max_gyro[i - 3] - global_min_gyro[i - 3])).astype(np.intc)

        ## Inicializo la variable y en 0 que me dice si hay patología o no
        y = 0

        ## Guardado de los escalogramas individuales
        np.savez_compressed(ruta_guardado_individual + nombre_base_segmento + '{}_00{}'.format(segment, i), y = y, X = X[i,:,:])

    ## Guardado de datos preprocesados en la ruta de salida
    np.savez_compressed(ruta_guardado + nombre_base_segmento + '{}'.format(segment), y = y, X = X)