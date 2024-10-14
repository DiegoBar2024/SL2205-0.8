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

def ValoresMagnetometro(ruta_muestra_cruda, ID_paciente, actividades):
    """
    Función que se encarga de extraer las mediciones del magnetómetro para un determinado paciente y un conjunto de actividades dado
    
    Parameters
    ----------
    ruta_muestra_cruda : str
        Es la ruta a la carpeta donde se encuentran todas las muestras crudas
    ID_paciente : int
        Es el ID del paciente cuyos datos se van a extraer
    actividad : list
        Es el conjunto de actividades que se van a procesar
    """

    ## En caso que se tenga una muestra larga
    if long_sample:

        ## <<directorio_muestra>> me guarda el directorio donde están los segmentos
        directorio_muestra = "/L%s" % (ID_paciente)

    ## En caso que se tenga una muestra corta
    else:

        ## <<directorio_muestra>> me guarda el directorio donde están los segmentos
        directorio_muestra = "/S%s" % (ID_paciente)

    ## Armo la ruta completa hacia la carpeta del paciente
    ruta_paciente = ruta_muestra_cruda + directorio_muestra
    
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
        dataframe = pd.read_csv(ruta_completa, skiprows = 1, sep = '\t')

        ## MEDIDAS DEL MAGNETÓMETRO EN EL EJE X
        ## Creo un dataframe que tenga una columna de nombre "Mag_x" pero que no tenga datos
        Mag_x = pd.DataFrame(columns = ["Mag_X"])

        ## Me quedo únicamente con aquella columna que tenga la lista de datos del magnetómetro en el eje x
        Mag_x = encontrar_columna(dataframe, "Mag_X", "Mag_X.1")

        ## MEDIDAS DEL MAGNETÓMETRO EN EL EJE Y
        ## Creo un dataframe que tenga una columna de nombre "Mag_y" pero que no tenga datos
        Mag_y = pd.DataFrame(columns = ["Mag_Y"])

        ## Me quedo únicamente con aquella columna que tenga la lista de datos del magnetómetro en el eje y
        Mag_y = encontrar_columna(dataframe, "Mag_Y", "Mag_Y.1")

        ## MEDIDAS DEL MAGNETÓMETRO EN EL EJE Z
        ## Creo un dataframe que tenga una columna de nombre "Mag_z" pero que no tenga datos
        Mag_z = pd.DataFrame(columns = ["Mag_Z"])

        ## Me quedo únicamente con aquella columna que tenga la lista de datos del magnetómetro en el eje z
        Mag_z = encontrar_columna(dataframe, "Mag_Z", "Mag_Z.1")

        ## Me quedo únicamente con aquella columna que tenga el nombre "Timestamp" o sea que contenga las muestras de tiempo
        Time = encontrar_columna(dataframe, "System_Timestamp_Plot_Zeroed", "Timestamp")

        ## DATOS FINALES
        ## Creo un nuevo dataframe y concateno todas los otros data frames (aceleraciones y giros) para formar la tabla final
        data_final = pd.DataFrame()
        data_final = pd.concat((Time, Mag_x, Mag_y, Mag_z), axis = 1, keys=("Time", "Mag_x", "Mag_y", "Mag_z"))

        ## Elimino los indices 0 y 1 (filas 0 y 1) de toda la tabla que me queda
        data_final.drop([0,1], axis = 0, inplace = True)

        ## Retorno los datos seleccionados
        return data_final

def encontrar_columna(dataframe, nombre1:str, nombre2:str):
    """
    Function to find the calibrated data corresponding to the desired column names.

    Parameters
    ----------
    dataframe: pd.DataFrame
    nombre1: str
    nombre2: str

    """
    ## Creo primero la plantilla tabular para los datos útiles que voy a extraer donde el nombre es una entrada
    ## Es importante que la creación del DataFrame quede por fuera del bucle for
    datos_utiles = pd.DataFrame(columns = [nombre1])

    ## La función <<dataframe.columns>> me da una "lista" del tipo pandas que tiene las cadenas con todos los nombres de mis columnas
    ## Como es una estructura iterable yo puedo ir elemento por elemento leyendo nombre por nombre de mi columna
    ## Ésto implica que el primer valor va a ser ACCEL_LN_X, el segundo ACCEL_LN_Y y así sucesivamente
    for columna in dataframe.columns:

        ## Lo que hago acá es setear una bandera a True en caso que el primer elemento de dicha columna (indice 0) sea CAL
        ## Dicho de otra manera estoy "seleccionando" únicamente aquellas columnas de datos que tengan CAL
        bandera = (dataframe[columna].iloc[0] == "CAL")

        ## Entro al if si se cumplen las siguientes cosas AL MISMO TIEMPO:
        ## i) La columna tiene que tener el valor CAL, lo cual se comprueba mirando el valor de <<bandera>>
        ## ii) Debe ocurrir que el nombre de la columna debe ser igual a <<nombre1>> o a <<nombre2>>
        ##     La comparación de los nombres se hace en minúscula para normalizar los caracteres
        ## En resumen yo me estoy quedando con aquellas columnas CAL y cuyo nombre sea igual a uno de los dos que pasé como parámetro de entrada
        if (bandera) and (columna.lower() == nombre1.lower() or columna.lower() == nombre2.lower()):
                
                ## Selecciono entonces la columna correspondiente en el dataframe que me interesa
                datos_utiles[nombre1] = dataframe[columna]

    ## Retorno un objeto DataFrame de Pandas ÚNICAMENTE con la columna que me interesa
    return datos_utiles[nombre1]

# # TEST
# datos_magnetometro = ValoresMagnetometro('C:/Yo/Tesis/sereData/sereData/Dataset/raw_2_process', 106, ['Caminando'])

# print(datos_magnetometro['Mag_x'])