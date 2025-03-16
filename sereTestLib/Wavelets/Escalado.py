####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import numpy as np
from math import ceil
import os, re
from copy import deepcopy
from natsort.natsort import natsorted
from parameters import dict_actividades
from Escalogramas import *

def Escalado(scalogram_path, output_path, escalado, patients = [], actividades = None, static_window_secs = 3, movil_window_secs = 3, overlap_secs = 0.5, fs = 200, escalogramas = None):
    """
    This function takes the created scalograms paste all together performs one or a combination of the next operations:
            Augment scalograms (windows_secs and overlap_secs controls it)
            Standarizes the scalogram or scales with pre fixed values.
            
    If the argument 'escalogramas' is None, the scalograms will be read in the 'scalogram_path'. 
    Else, that argument 'escalogramas' will be used instead.
    
    IMPORTANT: If the folder already exist, this function will not be executed
        
    Parameters
    ----------
    scalogram_path:  str
        Path to original segment's scalograms
    output_path: str 
        Destination  path
    escalado: str 
        Indicates whether the scalogram must be standarized or scaled with pre fixed values.   
    patients: list, optional 
        List of patiens. Defaults to [].
    actividades: list, optional
        List of activities to process. 
        The default is None, in which case all existing activities are processed
    static_window_secs: int, optional
        Window duration for de original segments. Defaults to 3.
    movil_window_secs: int, optional 
        Window duration for sample segment. Defaults to 3.
    overlap_secs: float, optional 
        Time that the output scalograms are overlaped, if overlap_secs is equal or greater than windows secs, there isn't overlap. Defaults to 0.5.
    fs: int, optional
        Sampling Frequency. Defaults to 200.
    escalogramas: list of dicts, optional
        List that in element 'i' there is a dictionary with the scalogram matrix of segment 'i' and the base name of that segment.
        The format of each dictionary is {'escalograma': array, 'nombre_base_segmento': str}.
        Default to None.
    """    
    # Esto esta raro, porque para aumentarlos, asumimos que son consecutivos. Pueden no serlo. Ver como arreglar. Capaz mirando el numero de segmento
    # Va a haber problema con la actividad si sumamos mas de 9 actividades (porque se busca que empiece con...1 ej y 10 tmbn empieza con 1)

    # TODO: Agregar en un log del usuario
    ## En caso de que el directorio no exista, lo creo
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #TODO: tener cuidado con que empiezan los archivos! Eso hay q modificarlo en createscalograms_cmorlet o en la clasificacion de estados
    #TODO: No funciona para crear de 5s, no crea todos los que tiene que crear.
    
    ## Por defecto se tiene que la variable <<escalogramas>> va a estar seteada a None
    ## De éste modo los escalogramas van a ser leídos desde la ruta <<scalogram_path>>
    if escalogramas is None:

        ## Se listan los ficheros dentro de la ruta <<scalogram_path>> y luego se ordenan de manera natural usando <<natsort>>
        files = natsorted(os.listdir(scalogram_path))

        ## <<actividades>> es una lista de cadenas cuyos elementos son las actividades que yo voy a querer procesar
        if actividades is not None:

            ## Tupla de actividades en numerico
            ## Se crea una tupla cuyos elementos son los valores asociados a las claves de las actividades que se pasan como entrada
            actividades = tuple([dict_actividades.get(actividad) for actividad in actividades])

            ## Selección de sub caso de actividad
            ## Asumo que el primer numero es la actividad (o el texto, veremos)
            ## Se listan únicamente aquellos archivos dentro del directorio que comiencen con alguno de los números asociados a las actividades que se van a procesar
            files = [file for file in files if file.startswith(actividades)]

        ## Total segments in the sample for the activty
        ## Para cada segmento procesado tengo 6 escalogramas, correspondientes a las 3 aceleraciones y a los 3 giros
        ## El valor <<segments>> me va a almacenar la cantidad de segmentos que se tienen en la carpeta
        segments = np.shape(files)[0] //6

    ## En caso de que escalogramas NO sea none
    else:
        segments = np.shape(escalogramas)[0]

    ## Size of the array needed to store the minimum segments in order to
    ## En caso de que el escalado sea "standarizado" (el que tomamos por defecto)
    if escalado == "standarizado":
        
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

            ## En caso que no haya colocado un diccionario de escalogramas a la entrada
            if escalogramas is None:

                ## <<indexes>> me va a almacenar los íncices de los 6 componentes del mismo segmento
                ## Recuerdo que cada segmento va a estar formado de 6 componentes que son los escalogramas de 
                indexes = np.arange(segment * 6, ((segment + 1) * 6)) #indexes of the 6 components of the same segment
                arch = [np.load(scalogram_path + files[ind]) for ind in indexes]
                X = [deepcopy(arch[j]['X']) for j in range(6)]
                X = np.dstack(X)
            
            ## En caso de que la cantidad de escalogramas a la entrada sea no nulo
            else:

                ## <<escalogramas>> es la lista de diccionarios cuyo i-esimo elemento es el diccionario asociado al i-ésimo segmento
                ## <<segment>> me da el identificador del segmento
                ## X va a ser el tensor tridimensional correspondiente al i-ésimo segmento (dimensiones frecuencia-tiempo-canales)
                X = escalogramas[segment]['escalograma']

            ## Hago el cálculo de los máximos y mínimos LOCALES AL ESCALOGRAMA ACTUAL
            ## Recuerdo que en cada escalograma tridimensional las tres capas superiores contienen los coeficientes de las aceleraciones
            ## Por otro lado las tres capas inferiores contienen los coeficientes de los giroscopios
            ## <<local_max_acc>> va a contener el elemento de mayor valor absoluto presente en cada matriz de coeficientes de wavelet de las aceleraciones
            ## Esto implica que <<local_max_acc>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada maximo
            local_max_acc =  [np.max(np.abs(X[:, :, 0])), np.max(np.abs(X[:, :, 1])), np.max(np.abs(X[:, :, 2]))]

            ## <<local_max_gyro>> va a contener el elemento de mayor valor absoluto presente en cada matriz de coeficientes de wavelet de los giroscopios
            ## Esto implica que <<local_max_gyro>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada maximo
            local_max_gyro = [np.max(np.abs(X[:, :, 3])),np.max(np.abs(X[:, :, 4])),np.max(np.abs(X[:, :, 5]))]

            ## <<local_min_acc>> va a contener el elemento de menor valor absoluto presente en cada matriz de coeficientes de wavelet de las aceleraciones
            ## Esto implica que <<local_min_acc>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada minimo
            local_min_acc =  [np.min(np.abs(X[:, :, 0])),np.min(np.abs(X[:, :, 1])),np.min(np.abs(X[:, :, 2]))]            

            ## <<local_min_gyro>> va a contener el elemento de menor valor absoluto presente en cada matriz de coeficientes de wavelet de los giroscopios
            ## Esto implica que <<local_min_gyro>> me va a quedar un vector de tres coordenadas donde cada coordenada es cada minimo
            local_min_gyro = [np.min(np.abs(X[:, :, 3])),np.min(np.abs(X[:, :, 4])),np.min(np.abs(X[:, :, 5]))]         

            ## Itero para cada una de las 3 coordenadas de los vectores anteriores
            ## Tengo 3 coordenadas porque tengo 3 canales de aceleración y 3 canales de giroscopios
            ## El propósito de ésto es actualizar los máximos y minimos valores de coeficientes globales en caso de que se vayan actualizando
            ## Con "coeficientes globales" me refiero a los coeficientes de TODOS los escalogramas de un mismo paciente
            ## Con "coeficientes locales" me refiero a los coeficientes del escalograma actual en la iteracion
            for i in range(3):

                ## En caso de que la i-ésima aceleración maxima del escalograma actual supere al valor global, lo actualizo
                if local_max_acc[i] > global_max_acc[i]:
                    global_max_acc[i] = local_max_acc[i]
                
                ## En caso de que el i-esimo valor de giroscopio maximo del escalograma actual supere al valor global, lo actualizo
                if local_max_gyro[i] > global_max_gyro[i]:
                    global_max_gyro[i] = local_max_gyro[i]
                
                ## En caso de que la i-ésima aceleración minima del escalograma actual sea inferior al valor global, lo actualizo
                if local_min_acc[i] < global_min_acc[i]:
                    global_min_acc[i] = local_min_acc[i]
                
                ## En caso de que el i-esimo valor de giroscopio minimo del escalograma actual sea inferior al valor global, lo actualizo
                if local_min_gyro[i] < global_min_gyro[i]:
                    global_min_gyro[i] = local_min_gyro[i]

    ## <<segments>> me da la cantidad de segmentos que tengo detectados
    ## Por defecto, tanto <<static_window_secs>> como <<movil_window_secs>> tienen un valor de 3 segundos y representa la cantidad de segundos de cada ventana
    ## Por defecto, <<overlap_secs>> tiene un valor de 0.5 segundos y representa la cantidad de segundos de solapamiento entre ventanas
    if segments * static_window_secs >= movil_window_secs:

            ## Itero para cada uno de los canales (tengo 6 canales que son tres aceleraciones y tres giroscopios)
            for i in range(X.shape[2]):
    
                ## Si el escalado es estandarizado (valor por defecto)
                if escalado == "standarizado":

                    ## En caso de que me esté refiriendo a las aceleraciones
                    if i < 3:

                        ## TRANSFORMACIÓN DE LOS ESCALOGRAMAS A INTENSIDADES DE GRIS
                        ## Xwin[:, :, i] la interpreto como la matriz tiempo-frecuencia en el i-ésimo canal
                        ## Xwin[:, :, i] tiene una forma (f, t) donde:
                        ## <<f>> me da la cantidad de frecuencias que tengo en el escalograma
                        ## <<t>> me da la cantidad de muestras temporales que tengo en el 
                        X[:, :, i] = np.rint((X[:, :, i] - global_min_acc[i]) * 256/(global_max_acc[i]-global_min_acc[i])).astype(np.intc)

                    ## En caso de que me esté refiriendo a los giroscopios
                    else:

                        ## TRANSFORMACIÓN DE LOS ESCALOGRAMAS A INTENSIDADES DE GRIS
                        ## Xwin[:, :, i] la interpreto como la matriz tiempo-frecuencia en el i-ésimo canal
                        ## Xwin[:, :, i] tiene una forma (f, t) donde:
                        ## <<f>> me da la cantidad de frecuencias que tengo en el escalograma
                        ## <<t>> me da la cantidad de muestras temporales que tengo en el escalograma
                        X[:, :, i] = np.rint((X[:, :, i] - global_min_gyro[i - 3]) * 256 / (global_max_gyro[i - 3] - global_min_gyro[i - 3])).astype(np.intc)

            ## Inicializo la variable y en 0 que me dice si hay patología
            y = 0

            ## Guardado de datos preprocesados en la ruta de salida
            np.savez_compressed(archivo_preprocesado + "_00" + i + '.npz', y = y, X = X)