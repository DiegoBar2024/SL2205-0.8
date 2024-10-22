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
from ValoresMagnetometro import ValoresMagnetometro

def GraficarDatosPreprocesados(ruta_muestra_preprocesada, ID_paciente, columna, actividades):
    '''
    Se grafican los datos preprocesados (aceleracion o giros) de un paciente (con una ID) para un conjunto dado de actividades
    
    Parameters
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
    
    Parameters
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

def GraficarAceleraciones(ruta_muestra_preprocesada, ID_paciente, actividades):
    '''
    Se grafican las tres aceleraciones de un paciente (con una ID) para un conjunto dado de actividades
    
    Parameters
    ----------
    ruta_muestra_preprocesada : str
        Es la ruta a la carpeta donde se encuentran todas las muestras preprocesadas
    ID_paciente : int
        Es el ID del paciente cuyos datos van a ser graficados
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

        ## Extraigo el vector de la aceleración en el eje x
        AC_x = np.array(dataframe['AC_x'])

        ## Extraigo el vector de la aceleración en el eje y
        AC_y = np.array(dataframe['AC_y'])

        ## Extraigo el vector de la aceleración en el eje z
        AC_z = np.array(dataframe['AC_z'])

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Grafico los datos. En mi caso las tres aceleraciones en el mismo eje
        plt.plot(tiempo, AC_x, color = 'r', label = '$a_x$')
        plt.plot(tiempo, AC_y, color = 'b', label = '$a_y$')
        plt.plot(tiempo, AC_z, color = 'g', label = '$a_z$')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Aceleracion $(m/s^2)$")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

def GraficarVelocidadesAngulares(ruta_muestra_preprocesada, ID_paciente, actividades):
    '''
    Se grafican las tres velocidades angulares de un paciente (con una ID) para un conjunto dado de actividades
    
    Parameters
    ----------
    ruta_muestra_preprocesada : str
        Es la ruta a la carpeta donde se encuentran todas las muestras preprocesadas
    ID_paciente : int
        Es el ID del paciente cuyos datos van a ser graficados
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

        ## Extraigo el vector de la velocidad angular en el eje x
        GY_x = np.array(dataframe['GY_x']) * np.pi / 180

        ## Extraigo el vector de la velocidad angular en el eje y
        GY_y = np.array(dataframe['GY_y']) * np.pi / 180

        ## Extraigo el vector de la velocidad angular en el eje z
        GY_z = np.array(dataframe['GY_z']) * np.pi / 180

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Grafico los datos. En mi caso las tres velocidades angulares
        plt.plot(tiempo, GY_x, color = 'r', label = '$w_x$')
        plt.plot(tiempo, GY_y, color = 'b', label = '$w_y$')
        plt.plot(tiempo, GY_z, color = 'g', label = '$w_z$')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la velocidad angular (rad/s)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Velocidad angular (rad/s)")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

def GraficarValoresMagnetometro(ruta_muestra_cruda, ID_paciente, actividades):
    '''
    Se grafican las tres señales tomadas del magnetómetro para un paciente dado y una conjunto de actividades específico
    
    Parameters
    ----------
    ruta_muestra_cruda : str
        Es la ruta a la carpeta donde se encuentran todas las muestras crudas
    ID_paciente : int
        Es el ID del paciente cuyos datos van a ser graficados
    actividad : list
        Es el conjunto de actividades que se van a procesar
    '''

    ## Se delega la responsabilidad de obtener los valores del magnetómetro a la función <<ValoresMagnetometro>>
    valores_magnetometro = ValoresMagnetometro(ruta_muestra_cruda, ID_paciente, actividades)

    ## Separo el vector de tiempos del dataframe
    tiempo = np.array(valores_magnetometro['Time'], dtype = float)

    ## Extraigo el vector de la medición del magnetómetro en el eje x 
    Mag_x = np.array(valores_magnetometro['Mag_x'])

    ## Extraigo el vector de la medición del magnetómetro en el eje y
    Mag_y = np.array(valores_magnetometro['Mag_y'])

    ## Extraigo el vector de la medición del magnetómetro en el eje z
    Mag_z = np.array(valores_magnetometro['Mag_z'])

    ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
    tiempo = (tiempo - tiempo[0]) / 1000

    ## Grafico los datos. En mi caso las tres medidas de magnetometros
    plt.plot(tiempo, Mag_x, color = 'r', label = '$mag_x$')
    plt.plot(tiempo, Mag_y, color = 'b', label = '$mag_y$')
    plt.plot(tiempo, Mag_z, color = 'g', label = '$mag_z$')

    ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la medida del magnetometro (flujo)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Medida magnetómetro (flujo local)")

    ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
    plt.legend()

    ## Despliego la gráfica
    plt.show()

#GraficarValoresMagnetometro('C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process', 10, ['Caminando'])

# TEST
## GraficarDatosPreprocesados('C:/Yo/Tesis/sereData/sereData/Dataset/dataset', 106, 'AC_x', ['Caminando'])
GraficarAceleraciones('C:/Yo/Tesis/sereData/sereData/Dataset/dataset', 278, ['Caminando'])

#GraficarVelocidadesAngulares('C:/Yo/Tesis/sereData/sereData/Dataset/dataset', 161, ['Caminando'])