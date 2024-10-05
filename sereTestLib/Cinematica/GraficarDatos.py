####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import parameters
from parameters import long_sample, dict_actividades
import os
from natsort.natsort import natsorted
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GraficarDatosEnteros(ruta_muestra_preprocesada, ID_paciente, columna, actividades):
    '''
    Se grafican los datos preprocesados (aceleracion o giros) de un paciente (con una ID) para un conjunto dado de actividades
    
    Parametros
    ----------
    ruta_muestra_preprocesada : str
        Es la ruta a la carpeta donde se encuentran todas las muestras preprocesadas
    ID_paciente : int
        Es el ID del paciente cuyos datos van a ser graficados
    columna : str
        Es el nombre de la columna cuyos datos van a graficarse
    actividad : list
        Es el conjunto de actividades que se van a procesar
    '''

    ## En caso que se tenga una muestra larga
    if long_sample:

        ## <<directorio_muestra>> me guarda el directorio donde están los segmentos
        directorio_muestra = "/L%s" % (ID_paciente)

    ## En caso que se tenga una muestra corta
    else:

        ## <<directorio_muestra>> me guarda el directorio donde están los segmentos
        directorio_muestra = "/S%s" % (ID_paciente)

    ## Armo la ruta completa hacia la carpeta del paciente
    ruta_paciente = ruta_muestra_preprocesada + directorio_muestra
    
    ## <<os.listdir>> lo que hace es listarme los directorios (o sea archivos y carpetas) que tengo en una determinada ruta pasada como parametro
    ## El comando natsorted lo que hace es ordenar un iterable de forma NATURAL
    fragmentos = natsorted(os.listdir(ruta_paciente))

    # Si se utiliza el argumento 'actividades', solo esas se preprocesan
    # En caso que el campo 'actividades' sea no nulo, yo quiero que únicamente éstas sean procesadas
    if actividades is not None:

        ## Se crea una tupla donde se guardan los valores de dict_actividades asociados a cada una de las actividades de la lista
        ## O sea que <<actividades>> me va a quedar una tupla cuyos elementos son los valores asociados a las actividades de la lista <<actividades>> en el diccionario <<dict_actividades>> 
        actividades = tuple([dict_actividades.get(actividad) for actividad in actividades])

        ## Lo que hago acá es generar una lista cuyos elementos sean el nombre de los fragmentos (ficheros) los cuales contengan datos de la(s) actividad(es) deseada(s)
        ## Según el caracter con el que comienza el nombre de un fragmento se hace referencia a una actividad en específico
        ## Las actividades se indexan de la siguiente manera:
        ## 1 -- Sentado
        ## 2 -- Parado
        ## 3 -- Caminando
        ## 4 -- Escalera
        fragmentos = [fragmento for fragmento in fragmentos if fragmento.startswith(actividades)]
    
    ## Itero para cada uno de los .csv que seleccioné antes
    for fichero in fragmentos:

        ## Genero la ruta completa del .csv
        ruta_completa = ruta_paciente + '/' + fichero

        ## Se el archivo .csv y se almacenan los datos en un DataFrame Pandas
        dataframe = pd.read_csv(ruta_completa)

        ## Separo el vector de tiempos del dataframe
        tiempo = np.array(dataframe['Time'])

        ## Separo la columna de datos de interés dado como parámetro de entrada
        datos_a_graficar = np.array(dataframe[columna])

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Grafico los datos
        plt.plot(tiempo, datos_a_graficar)
        plt.show()

def GraficarDatosSegmentados(ruta_muestra_segmentada, ID_paciente, columna, actividades):
    '''
    Se grafican todos los datos segmentados (aceleracion o giros) de un paciente (con una ID) para un conjunto dado de actividades
    
    Parametros
    ----------
    ruta_muestra_segmentada : str
        Es la ruta a la carpeta donde se encuentran todas las muestras segmentadas
    ID_paciente : int
        Es el ID del paciente cuyos datos van a ser graficados
    columna : str
        Es el nombre de la columna cuyos datos van a graficarse
    actividad : list
        Es el conjunto de actividades que se van a procesar
    '''

    ## En caso que se tenga una muestra larga
    if long_sample:

        ## <<directorio_muestra>> me guarda el directorio donde están los segmentos
        directorio_muestra = "/L%s" % (ID_paciente)

    ## En caso que se tenga una muestra corta
    else:

        ## <<directorio_muestra>> me guarda el directorio donde están los segmentos
        directorio_muestra = "/S%s" % (ID_paciente)

    ## Armo la ruta completa hacia la carpeta del paciente
    ruta_paciente = ruta_muestra_segmentada + directorio_muestra
    
    ## <<os.listdir>> lo que hace es listarme los directorios (o sea archivos y carpetas) que tengo en una determinada ruta pasada como parametro
    ## El comando natsorted lo que hace es ordenar un iterable de forma NATURAL
    fragmentos = natsorted(os.listdir(ruta_paciente))

    # Si se utiliza el argumento 'actividades', solo esas se preprocesan
    # En caso que el campo 'actividades' sea no nulo, yo quiero que únicamente éstas sean procesadas
    if actividades is not None:

        ## Se crea una tupla donde se guardan los valores de dict_actividades asociados a cada una de las actividades de la lista
        ## O sea que <<actividades>> me va a quedar una tupla cuyos elementos son los valores asociados a las actividades de la lista <<actividades>> en el diccionario <<dict_actividades>> 
        actividades = tuple([dict_actividades.get(actividad) for actividad in actividades])

        ## Lo que hago acá es generar una lista cuyos elementos sean el nombre de los fragmentos (ficheros) los cuales contengan datos de la(s) actividad(es) deseada(s)
        ## Según el caracter con el que comienza el nombre de un fragmento se hace referencia a una actividad en específico
        ## Las actividades se indexan de la siguiente manera:
        ## 1 -- Sentado
        ## 2 -- Parado
        ## 3 -- Caminando
        ## 4 -- Escalera
        fragmentos = [fragmento for fragmento in fragmentos if fragmento.startswith(actividades)]
    
    ## Itero para cada uno de los .csv que seleccioné antes
    for fichero in fragmentos:

        ## Genero la ruta completa del .csv
        ruta_completa = ruta_paciente + '/' + fichero

        ## Se el archivo .csv y se almacenan los datos en un DataFrame Pandas
        dataframe = pd.read_csv(ruta_completa)
    
        ## Separo el vector de tiempos del dataframe
        tiempo = np.array(dataframe['Time'])

        ## Separo la columna de datos de interés dado como parámetro de entrada
        datos_a_graficar = np.array(dataframe[columna])

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Grafico los datos
        plt.plot(tiempo, datos_a_graficar)
        plt.show()

GraficarDatosEnteros('C:/Yo/Tesis/sereData/sereData/Dataset/dataset', 106, 'AC_x', ['Caminando'])
