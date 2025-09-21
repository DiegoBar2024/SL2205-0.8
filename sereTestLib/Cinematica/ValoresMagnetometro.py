## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from parameters import *
from Muestreo import *
import parameters
from parameters import long_sample, dict_actividades
import os
from natsort.natsort import natsorted
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as it

## ----------------------------------- OBTENCIÓN VALORES MAGNETOMETRO ----------------------------------

def ValoresMagnetometro(id_persona = None, lectura_datos_propios = False, ruta = None):

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
    columnas_deseadas = ['Time', 'Mag_x', 'Mag_y', 'Mag_z']

    ## Creo un diccionario con los nombres originales de las columnas y sus nombres nuevos
    nombres_columnas = {'Timestamp': 'Time', 'Mag_X_CAL' : 'Mag_x', 'Mag_Y_CAL' : 'Mag_y', 'Mag_Z_CAL' : 'Mag_z'}

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
    mag = np.array([np.array(data['Mag_x']), np.array(data['Mag_y']), np.array(data['Mag_z'])]).transpose()
    
    ## Retorno los resultados al realizar la lectura correspondiente
    return mag

## Ejecución principal del programa
if __name__== '__main__':

    ## Especifico la ruta en la cual se encuentra el registro a leer
    ruta_registro_completa = ruta_registro + 'Actividades_Sabrina.txt'

    ## Hago la lectura de los datos
    mag = ValoresMagnetometro(id_persona = 69, lectura_datos_propios = True, ruta = ruta_registro_completa)