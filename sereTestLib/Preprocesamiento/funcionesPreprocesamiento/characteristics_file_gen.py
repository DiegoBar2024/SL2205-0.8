####### IMPORTO LA CARPETA DONDE ESTÁN LOS PARÁMETROS ########
import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
##############################################################

import numpy as np
import pandas as pd
import os
import re
from natsort.natsort import natsorted
from scipy import signal

def characteristics_file_gen(directorio_segmentos_muestra, directorio_caracteristicas, filter_window=5):
    """
    Generates the characteristics file by parsing the raw data file.
    Also splits the raw acquisition file in time_frame sub-samples

    Parameters
    ----------
    directorio_segmentos_muestra : str
       Path to the raw data file

    directorio_caracteristicas : str
        Path to the characteristics file.

    filter_window : int
        Window for median filter. Size in samples
    """
    
    ## La variable <<estado>> indica el estado de actividad
    ## Dicho estado sigue el siguiente código: (0 - Sit, 1 - Rise, 2 - Walk, 3 - Climb, 99 - Unknown ) 
    estado = 99

    # Crear directorio salida caracteristicas
    if not os.path.exists(directorio_caracteristicas):   #TODO: Agregar en un log del usuario
        os.makedirs(directorio_caracteristicas)

    #TODO: Es ver como hacer para no guardar un archivo por segmento. Si es que ocupan mucho
    archivos_segmentos = natsorted(os.listdir(directorio_segmentos_muestra))

    # busco el tipo de adquisición S para corta y L para larga
    tipo_adquisicion = re.split(r'\d',archivos_segmentos[0])[not""]
    nombre_base = re.split(r'\D',archivos_segmentos[0])

    ## Defino una lista con las columnas que yo voy a querer en el archivo de salida
    cl = ['meanAC_x', 'meanAC_y', 'meanAC_z', 'stdAC_x', 'stdAC_y', 'stdAC_z', 'state', 'file_name']

    ## Defino características como un DataFrame inicialmente vacío donde las columnas son las que están en la lista anterior
    caracteristicas = pd.DataFrame(columns = cl)

    k = 0

    ## Defino el nombre de mi archivo de salida
    archivo_salida = directorio_caracteristicas + tipo_adquisicion.upper()+ nombre_base[1] + '.csv'

    ## Itero para cada uno de los segmentos en la lista de segmentos
    for archivo_segmento in archivos_segmentos:

        ## Completo la ruta del archivo de entrada
        archivo_entrada = directorio_segmentos_muestra + archivo_segmento

        ## Extraigo el nombre base del archivo de segmento elminando del nombre la extensión csv
        nombre_base_segmento = archivo_segmento.replace('.csv','')

        ## Hago la lectura del segmento como csv
        segmento = pd.read_csv(archivo_entrada, low_memory=False)

        ## Aplico un filtro de mediana al segmento con una ventana de tamaño <<filter_window>> que paso como parámetro de entrada
        filt_splited_data = medianFilter(segmento, filter_window)

        ## filt_splited_data.set_axis(["Time", "AC_x", "AC_y", "AC_z", "GY_x", "GY_y", "GY_z"], axis=1, inplace=True)
        filt_splited_data.set_axis(["Time", "AC_x", "AC_y", "AC_z", "GY_x", "GY_y", "GY_z"], axis=1)

        if tipo_adquisicion.upper() == 'S':
            estado, n_paciente, n_segmento = re.split(r'\D',nombre_base_segmento)
            params = calc_and_load(filt_splited_data, estado, archivo_segmento)
            caracteristicas.loc[k] = params
            k += 1
        else:
            params = calc_and_load(filt_splited_data, estado, archivo_segmento)
            caracteristicas.loc[k] = params
            k += 1

    caracteristicas.to_csv(archivo_salida, index=False)

def calc_and_load(signal, state, data_file=""):
    """
    Process and returns the characteristics parameters for each frame of the raw acquisition sample.

    Parameters
    ----------
    signal : ndarray
        Data frame
    state : int
        Indicates the activity state (0 - Sit, 1 - Rise, 2 - Walk, 3 - Climb, 99 - Unknown )
    data_file : str
            Path to file with signal data

    age: 0 and patology 10 means no data was provided
    """

    ## Calculo los valores medios de las aceleraciones en los tres ejes que paso como parámetro
    meanAC_x, meanAC_y, meanAC_z = meansAceleration(signal)

    ## Calculo las desviaciones estándar de las aceleraciones en los tres ejes que paso como parámetro
    stdAC_x, stdAC_y, stdAC_z = stdAceleration(signal)

    return meanAC_x, meanAC_y, meanAC_z, stdAC_x, stdAC_y, stdAC_z, int(state), data_file

## Tengo como entrada un objeto DataFrame y un tamaño de ventana
## Como salida retorno un DataFrame el cual sea la versión filtrada de la entrada usando el filtro de mediana con el tamaño de ventana que pasé como parámetro
def medianFilter (data_frame, window):

    ## Creo en primer lugar un DataFrame vacío en principio sin ninguna columna
    dataframe_filtrado = pd.DataFrame()

    ## Itero columna por columna para el DataFrame de entrada
    for elem in data_frame.columns:
        
        ## REVISAR LAS SIGUIENTES INSTRUCCIONES PORQUE NO FUNCIONAN BIEN OPERANDO DIRECTAMENTE SOBRE UN DATAFRAME PANDAS
        ## LO QUE SI FUNCIONA ES TRADUCIR EL DATAFRAME A UN NUMPY ARRAY Y LUEGO CALCULAR LOS VALORES ESTADÍSTICOS DE AHI

        ## Se aplica un filtro de mediana al vector de datos de entrada
        ## Éste filtro lo que hace es primero hacer un padding de ceros al vector de entrada
        ## Luego toma una ventana (en éste caso unidimensional) de tamaño <<window>> y la va deslizando por todo el vector
        ## Tomando el índice de salida como el que está parado, calcula la mediana para ese vector en esa ventana
        columna_filtrada = signal.medfilt(data_frame[elem], window)

        ## Creo un nuevo DataFrame el cual contenga la columna filtrada
        columna_filtrada = pd.DataFrame(columna_filtrada, columns = [elem])

        ## Concateno la columna que filtré con el DataFrame de las columnas que ya fui filtrando
        dataframe_filtrado = pd.concat((dataframe_filtrado, columna_filtrada), axis = 1)

    return dataframe_filtrado

## REVISAR
## Funcion al que entra el DataFrame del segmento preprocesado y salen los valores medios para las aceleraciones en los tres ejes
def meansAceleration(df):
    """
    Calculate the acceleration mean value per axis

    Parameters
    ----------
    df

    Returns
    -------
    meanAC_x : int
        Acceleration mean value in the data axis
    meanAC_y : int
        Acceleration mean value in the y axis
    meanAC_z : int
        Acceleration mean value in the z axis
    """

    ## REVISAR LAS SIGUIENTES INSTRUCCIONES PORQUE NO FUNCIONAN BIEN OPERANDO DIRECTAMENTE SOBRE UN DATAFRAME PANDAS
    ## LO QUE SI FUNCIONA ES TRADUCIR EL DATAFRAME A UN NUMPY ARRAY Y LUEGO CALCULAR LOS VALORES ESTADÍSTICOS DE AHI
    ## HAGO PROPOSICIONES ALTERNATIVAS PARA CADA UNA DE LAS LÍNEAS

    ## Se calcula el valor medio de los valores de aceleración en el eje x
    ## PROPOSICIÓN ALTERNATIVA: 
    ##meanAC_x = df["AC_x"].to_numpy(dtype=float).mean() 
    meanAC_x = df["AC_x"].mean()

    ## Se calcula el valor medio de los valores de aceleración en el eje y
    ## PROPOSICIÓN ALTERNATIVA: 
    ## meanAC_y = df["AC_y"].to_numpy(dtype=float).mean() 
    meanAC_y = df["AC_y"].mean()

    ## Se calcula el valor medio de los valores de aceleración en el eje z
    ## PROPOSICIÓN ALTERNATIVA: 
    ## meanAC_z = df["AC_z"].to_numpy(dtype=float).mean() 
    meanAC_z = df["AC_z"].mean()

    return meanAC_x, meanAC_y, meanAC_z

## REVISAR
## Funcion al que entra el DataFrame del segmento preprocesado y salen las desviaciones estándar para las aceleraciones en los tres ejes
def stdAceleration(df):
    """
    Calculate the acceleration standard deviation value per axis

    Parameters
    ----------
    df

    Returns
    -------
    stdAC_x : int
        Acceleration standard deviation value in the data axis
    stdAC_y : int
        Acceleration standard deviation value in the y axis
    stdAC_z : int
        Acceleration standard deviation value in the z axis
    """

    ## REVISAR LAS SIGUIENTES INSTRUCCIONES PORQUE NO FUNCIONAN BIEN OPERANDO DIRECTAMENTE SOBRE UN DATAFRAME PANDAS
    ## LO QUE SI FUNCIONA ES TRADUCIR EL DATAFRAME A UN NUMPY ARRAY Y LUEGO CALCULAR LOS VALORES ESTADÍSTICOS DE AHI
    ## HAGO PROPOSICIONES ALTERNATIVAS PARA CADA UNA DE LAS LÍNEAS

    ## Se calcula la desviación estándar de los valores de aceleración en el eje x
    ## PROPOSICIÓN ALTERNATIVA: 
    ## stdAC_x = df["AC_x"].to_numpy(dtype=float).std() 
    stdAC_x = df["AC_x"].std()

    ## PROPOSICIÓN ALTERNATIVA: 
    ## stdAC_y = df["AC_y"].to_numpy(dtype=float).std() 
    ## Se calcula la desviación estándar de los valores de aceleración en el eje y
    stdAC_y = df["AC_y"].std()

    ## PROPOSICIÓN ALTERNATIVA: 
    ## stdAC_z = df["AC_z"].to_numpy(dtype=float).std() 
    ## Se calcula la desviación estándar de los valores de aceleración en el eje z
    stdAC_z = df["AC_z"].std()

    return stdAC_x, stdAC_y, stdAC_z
