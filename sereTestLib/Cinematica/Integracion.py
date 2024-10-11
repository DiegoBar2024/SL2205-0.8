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
import scipy.integrate as it

def CalcularVelocidades(ruta_muestra_preprocesada, ID_paciente, actividades):
    '''
    Se realiza la integración de la señal de los acelerómetros para obtener las velocidades en los tres ejes
    En ésta función NO se hace ningún tipo de compensación de deriva. Se usa método trapezoidal.
    
    Parametros
    ----------
    ruta_muestra_preprocesada : str
        Es la ruta a la carpeta donde se encuentran todas las muestras preprocesadas
    ID_paciente : int
        Es el ID del paciente cuyos datos se van a extraer
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

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Extraigo el vector de la aceleración en el eje x
        AC_x = np.array(dataframe['AC_x'])

        ## Se calcula la integral en el tiempo usando el método trapezoidal asuminendo condiciones iniciales nulas
        v_x = it.cumulative_trapezoid(AC_x, tiempo, initial = 0)

        ## Extraigo el vector de la aceleración en el eje y
        AC_y = np.array(dataframe['AC_y'])
        
        ## Se calcula la integral en el tiempo usando el método trapezoidal asumiendo condiciones iniciales nulas
        v_y = it.cumulative_trapezoid(AC_y, tiempo, initial = 0)

        ## Extraigo el vector de la aceleración en el eje z
        AC_z = np.array(dataframe['AC_z'])

        ## Se calcula la integral en el tiempo usando el método trapezoidal asumiendo condiciones iniciales nulas
        v_z = it.cumulative_trapezoid(AC_z, tiempo, initial = 0)

        ## Grafico los datos. En mi caso las tres velocidades en el mismo par de ejes
        plt.plot(tiempo, v_x, color = 'r', label = '$v_x$')
        plt.plot(tiempo, v_y, color = 'b', label = '$v_y$')
        plt.plot(tiempo, v_z, color = 'g', label = '$v_z$')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Velocidad (m/s)")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

        ## Retorno una tupla con los vectores correspondientes a las tres velocidades
        return (v_x, v_y, v_z)

def CalcularPosiciones(ruta_muestra_preprocesada, ID_paciente, actividades):
    '''
    Se realiza la doble integración de la señal de los acelerómetros para obtener las posiciones en los tres ejes
    En ésta función NO se hace ningún tipo de compensación de deriva. Se usa método trapezoidal.
    
    Parametros
    ----------
    ruta_muestra_preprocesada : str
        Es la ruta a la carpeta donde se encuentran todas las muestras preprocesadas
    ID_paciente : int
        Es el ID del paciente cuyos datos se van a extraer
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

        ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
        tiempo = (tiempo - tiempo[0]) / 1000

        ## Extraigo el vector de la aceleración en el eje x
        AC_x = np.array(dataframe['AC_x'])

        ## Se calcula la integral en el tiempo usando el método trapezoidal para obtener la velocidad asuminendo condiciones iniciales nulas
        v_x = it.cumulative_trapezoid(AC_x, tiempo, initial = 0)
        
        ## Se calcula la integral en el tiempo usando el método trapezoidal para obtener la posicion asuminendo condiciones iniciales nulas
        x_x = it.cumulative_trapezoid(v_x, tiempo, initial = 0)

        ## Extraigo el vector de la aceleración en el eje y
        AC_y = np.array(dataframe['AC_y'])
        
        ## Se calcula la integral en el tiempo usando el método trapezoidal asumiendo condiciones iniciales nulas
        v_y = it.cumulative_trapezoid(AC_y, tiempo, initial = 0)

        ## Se calcula la integral en el tiempo usando el método trapezoidal para obtener la posicion asuminendo condiciones iniciales nulas
        x_y = it.cumulative_trapezoid(v_y, tiempo, initial = 0)

        ## Extraigo el vector de la aceleración en el eje z
        AC_z = np.array(dataframe['AC_z'])

        ## Se calcula la integral en el tiempo usando el método trapezoidal asumiendo condiciones iniciales nulas
        v_z = it.cumulative_trapezoid(AC_z, tiempo, initial = 0)

        ## Se calcula la integral en el tiempo usando el método trapezoidal para obtener la posicion asuminendo condiciones iniciales nulas
        x_z = it.cumulative_trapezoid(v_z, tiempo, initial = 0)

        ## Grafico los datos. En mi caso las tres velocidades en el mismo par de ejes
        plt.plot(tiempo, x_x, color = 'r', label = '$x_x$')
        plt.plot(tiempo, x_y, color = 'b', label = '$x_y$')
        plt.plot(tiempo, x_z, color = 'g', label = '$x_z$')

        ## Nomenclatura de ejes. En el eje x tenemos el tiempo (s) y en el eje y la aceleracion (m/s2)
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Posición (m)")

        ## Agrego la leyenda para poder identificar que curva corresponde a cada aceleración
        plt.legend()

        ## Despliego la gráfica
        plt.show()

        ## Retorno una tupla con los vectores correspondientes a las tres velocidades
        return (x_x, x_y, x_z)

## CalcularVelocidades('C:/Yo/Tesis/sereData/sereData/Dataset/dataset', 106, ['Caminando'])