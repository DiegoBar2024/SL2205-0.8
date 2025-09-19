## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
import parameters
from parameters import long_sample, dict_actividades
import os
from natsort.natsort import natsorted
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as it

## ----------------------------------- OBTENCIÓN VALORES MAGNETOMETRO ----------------------------------

def LecturaDatos(id_persona = None, lectura_datos_propios = False, ruta = None):

    ## --- Sistema de coordenadas usados en las medidas por los estudiantes ----------
    ##          EJE X: MEDIOLATERAL CON EJE POSITIVO A LA IZQUIERDA
    ##          EJE Y: VERTICAL CON EJE POSITIVO HACIA ARRIBA
    ##          EJE Z: ANTEROPOSTERIOR CON EJE POSITIVO HACIA DELANTE
    ## --- Los tres ejes de coordenadas deben formar una terna directa de vectores ---
    ## -------------------------------------------------------------------------------

    ## Abro el fichero correspondiente
    fichero = open(ruta, "r")

    ## Hago la lectura de todas las lineas correspondientes al fichero
    lineas = fichero.readlines()

    ## Creo un array vacío en donde voy a guardar los datos
    data = []

    ## Itero para todas aquellas lineas que tengan información útil
    for linea in lineas[3:]:

        ## Hago la traducción de la línea de datos a una lista de numeros flotantes, segmentando la línea por tabulación
        lista_datos = list(map(float,linea.split("\t")[:-1]))

        ## Agrego la lista de datos como renglón de la matriz de datos
        data.append(lista_datos)

    ## Hago una lista con todos los headers de los datos tomados
    headers = lineas[1].split("\t")[:-1]

    ## Hago el pasaje de los datos en forma de matriz a forma de dataframe
    data = pd.DataFrame(data, columns = headers)

    ## Creo una lista con las columnas deseadas
    columnas_deseadas = ['Time', 'AC_x', 'AC_y', 'AC_z', 'GY_x', 'GY_y', 'GY_z']

    ## Creo un diccionario con los nombres originales de las columnas y sus nombres nuevos
    nombres_columnas = {'Timestamp': 'Time', 'Accel_LN_X_CAL' : 'AC_x', 'Accel_LN_Y_CAL' : 'AC_y', 'Accel_LN_Z_CAL' : 'AC_z'
                        ,'Gyro_X_CAL' : 'GY_x', 'Gyro_Y_CAL' : 'GY_y', 'Gyro_Z_CAL' : 'GY_z'}

    ## Itero para cada una de las columnas del dataframe
    for columna in data.columns:

        ## Itero para cada uno de los nombres posibles
        for nombre in nombres_columnas.keys():

            ## En caso de que un nombre esté en la columna
            if nombre in columna:

                ## Renombro la columna
                data = data.rename(columns = {columna : nombres_columnas[nombre]})

    ## Selecciono las columnas deseadas
    data = data[columnas_deseadas]

    ## Armamos una matriz donde las columnas sean las aceleraciones
    acel = np.array([np.array(data['AC_x']), np.array(data['AC_y']), np.array(data['AC_z'])]).transpose()

    ## Armamos una matriz donde las columnas sean los valores de los giros
    gyro = np.array([np.array(data['GY_x']), np.array(data['GY_y']), np.array(data['GY_z'])]).transpose()

    ## Separo el vector de tiempos del dataframe
    tiempo = np.array(data['Time'])

    ## Se arma el vector de tiempos correspondiente mediante la traslación al origen y el escalamiento
    tiempo = (tiempo - tiempo[0]) / 1000

    ## Cantidad de muestras de la señal
    cant_muestras = len(tiempo)

    ## En caso de que no haya un período de muestreo bien definido debido al vector de tiempos de la entrada
    if all([x == True for x in np.isnan(tiempo)]):

        ## Asigno arbitrariamente una frecuencia de muestreo de 200Hz es decir período de muestreo de 0.005s
        periodoMuestreo = 0.005

    ## En caso de que el vector de tiempos contenga elementos numéricos
    else:

        ## Calculo el período de muestreo en base al vector correspondiente
        periodoMuestreo = PeriodoMuestreo(data)
    
    ## Retorno los resultados al realizar la lectura correspondiente
    return data, acel, gyro, cant_muestras, periodoMuestreo, tiempo