## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import pywt
import pandas as pd
import json
from scipy.signal import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')

from LecturaDatos import *
from LecturaDatosPacientes import *
from Fourier import *

def EnergiaDWT(señal, periodoMuestreo):

    ## ---------------------------------------- CÁLCULO DE ENERGÍA -----------------------------------------

    ## Descomposición multinivel usando DWT
    coefs = pywt.wavedec(data = señal, wavelet = 'dmey', mode = 'periodization', level = 8)

    ## Seteo la variable donde guardo el valor de la energía en 0
    energia_multinivel = 0

    ## Creo una lista donde asocio los coeficientes con las sub bandas de frecuencia
    subbandas = []

    ## Itero para cada una de las listas de coeficientes
    for i in range (1, len(coefs)):

        ## Actualizo el valor de la energía agregando el total de los cuadrados de los coeficientes
        energia_multinivel += np.sum(np.square(coefs[i]))

        ## Especifico el rango de frecuencias correspondiente a la subbanda
        rango = [2 ** ( - (i + 1)) / periodoMuestreo, 2 ** ( - i ) / periodoMuestreo]

        ## Asocio los coeficientes con la subbanda correspondiente y su energia
        subbandas.append((rango, np.sum(np.square(coefs[len(coefs) - i])), coefs[len(coefs) - i]))

    ## Asigno el rango correspondiente a los coefs de aproximación
    rango = [0, 2 ** ( - (len(coefs))) / periodoMuestreo]

    ## Sumo la energia correspodniente a la aproximación
    energia_multinivel += np.sum(np.square(coefs[0]))

    ## Agrego los coeficientes de aproximación asociandolos con sus bandas de frecuencia y su energía
    subbandas.append((rango, np.sum(np.square(coefs[0])), coefs[0]))

    ## Creo una lista donde guardo los rangos
    rangos = []

    ## Creo una lista donde guardo las energias
    energias = []

    ## Itero para cada una de las bandas correspondientes de la descomposición
    for banda in subbandas[::-1]:

        ## Agrego el rango a la lista de energía
        rangos.append("{} - {}".format(round(banda[0][0], 3), round(banda[0][1], 3)))

        ## Agrego la energia relativa a la lista de energias
        energias.append(banda[1] / energia_multinivel)

    ## Retorno la distribución de energias en subbandas
    return energias

        # ## Hago el gráfico de barras correspondiente
        # plt.bar(np.array(rangos), np.array(energias))
        # plt.xticks(rotation = 10)
        # plt.xlabel("Frecuencias (Hz)")
        # plt.ylabel("Energía Relativa")
        # plt.show()

## Rutina principal del programa
if __name__== '__main__':

    ## Inicializo una bandera la cual me permita distinguir si quiero guardar energías o procesarlas
    guardar = False
    
    ## Hago la lectura de los datos generales de los pacientes
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## En caso de que quiera generar y guardar la base de datos con las energías de los pacientes en cada subbanda
    if guardar:

        ## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
        for id_persona in ids_existentes:

            ## Coloco un bloque try en caso de que ocurra algún error de procesamiento
            try:

                ## Construyo una lista donde voy a guardar las listas de las energías relativas para cada uno de los canales
                energias_canales = []

                ## Hago la lectura de los datos del registro de marcha del paciente
                data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona, lectura_datos_propios = False)

                ## Itero para cada una de las tres señales de aceleraciones y giroscopios
                for i in range (3):

                    ## Hago la descomposición de energía en subbandas de la señal del i-ésimo acelerómetro usando DWT
                    energias_ac = EnergiaDWT(acel[:, i], periodoMuestreo)

                    ## Hago la descomposición de energía en subbandas de la señal del i-ésimo acelerómetro usando DWT
                    energias_gy = EnergiaDWT(gyro[:, i], periodoMuestreo)

                    ## Agrego el valor de la energía de la señal de acelerómetro
                    energias_canales.append(energias_ac)

                    ## Agrego el valor de la energía de la señal del giroscopio
                    energias_canales.append(energias_gy)

                ## Hago la lectura del archivo JSON de energías previamente existente
                with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias_DWT.json", 'r') as openfile:

                    # Cargo el diccionario el cual va a ser un objeto JSON
                    dicc_energias = json.load(openfile)

                ## Agrego en el diccionario los datos de energías del paciente actual
                dicc_energias[str(id_persona)] = energias_canales

                ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir el diccionario de energías
                with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias_DWT.json", "w") as outfile:

                    ## Escribo el diccionario actualizado
                    json.dump(dicc_energias, outfile)
        
            ## Si hay un error de procesamiento
            except:

                ## Que siga a la siguiente muestra
                continue
    
    ## En caso de que quiera procesar el JSON de energías y no guardarlo
    else:

        ## Hago la lectura del archivo JSON de energías previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias_DWT.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            dicc_energias = json.load(openfile)
        
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

        ## Genero un diccionario conteniendo los valores de energía en subbandas de aquellos pacientes que están etiquetados estables
        energias_etiquetados_estables = {}

        ## Genero un diccionario conteniendo los valores de energía en subbandas de aquellos pacientes que están etiquetados inestables
        energias_etiquetados_inestables = {}

        ## Itero para cada una de las IDs que están etiquetadas y son estables
        for id in id_estables:

            ## En caso de que la ID exista en el diccionario
            if str(id) in list(dicc_energias.keys()):

                ## Actualizo el diccionario (la key tiene que ser cadena)
                energias_etiquetados_estables[str(id)] = dicc_energias[str(id)]
        
        ## Itero para cada una de las IDs que están etiquetadas y son inestables
        for id in id_inestables:

            ## En caso de que la ID exista en el diccionario
            if str(id) in list(dicc_energias.keys()):

                ## Actualizo el diccionario (la key tiene que ser cadena)
                energias_etiquetados_inestables[str(id)] = dicc_energias[str(id)]
        
        ## Inicializo una matriz vacía donde guardo los valores medios de las energías en las subbandas para estables
        energia_media_estables = np.zeros((6, 9))

        ## Inicializo una lista vacía donde guardo los valores medios de las energías en las subbandas para inestables
        energia_media_inestables = np.zeros((6, 9))

        ## Itero para cada uno de los pacientes estables etiquetados
        for id in list(energias_etiquetados_estables.keys()):

            ## Actualizo el vector de energías sumando la energía del paciente
            energia_media_estables = np.add(energia_media_estables, np.array(energias_etiquetados_estables[id]))
        
        ## Itero para cada uno de los pacientes inestables etiquetados
        for id in list(energias_etiquetados_inestables.keys()):

            ## Actualizo el vector de energías sumando la energía del paciente
            energia_media_inestables = np.add(energia_media_inestables, np.array(energias_etiquetados_inestables[id]))
        
        ## Hago la normalización para obtener la energía promedio en el caso de estables
        energia_media_estables = energia_media_estables / len(energias_etiquetados_estables)

        ## Hago la normalización para obtener la energía promedio en el caso de inestables
        energia_media_inestables = energia_media_inestables / len(energia_media_inestables)

        ## Creo una lista en la cual voy asignando cada uno de los rangos de frecuencias
        rangos = []

        ## Itero para cada una de las listas de coeficientes
        for i in range (1, 9):

            ## Especifico el rango de frecuencias correspondiente a la subbanda
            rango = [2 ** ( - (i + 1)) / 0.005, 2 ** ( - i ) / 0.005]

            ## Agrego el rango a la lista de rangos
            rangos.append("{} - {}".format(2 ** ( - (i + 1)) / 0.005, 2 ** ( - i ) / 0.005))

        ## Asigno el rango correspondiente a los coefs de aproximación
        rango = rangos.append("{} - {}".format(0, 2 ** ( - 9) / 0.005))

        rangos.reverse()

        ## Especifico los títulos de los gráficos ordenados según la aparición
        titulos = ['$AC_{x}$', '$GY_{x}$', '$AC_{y}$', '$GY_{y}$', '$AC_{z}$', '$GY_{z}$']

        ## Itero para cada uno de los seis canales
        for i in range (6):

            ## Seteo las dimensiones del gráfico de barras
            w, x = 0.4, np.arange(9)

            ## Hago la graficación de la energía en las subbandas
            figure, axis = plt.subplots()
            axis.bar(x - w/2, energia_media_estables[i, :] / np.sum(energia_media_estables[i, :]), width = w, label = 'Estables')
            axis.bar(x + w/2, energia_media_inestables[i, :] / np.sum(energia_media_inestables[i, :]), width = w, label = 'Inestables')
            axis.set_xticks(x, rangos, rotation = 10)
            axis.set_title(titulos[i])
            axis.set_xlabel("Frecuencias (Hz)")
            axis.set_ylabel("Energía Relativa")
            axis.legend()
            plt.show()