## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import os
import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path().resolve()).replace('\\','/') + '/sereTestLib')
from parameters import *

## ----------------------------------- LECTURA DE DATOS DEL PACIENTE --------------------------------------

## Construyo una función la cual me permita leer los datos de los pacientes que están ingresados en la base de datos
def LecturaDatosPacientes():

    ## ----------------------------------------- LECTURA DE DATOS ------------------------------------------

    ## Hago la lectura del archivo .csv y lo convierto en un dataframe pandas
    datos_pacientes = pd.read_csv(ruta_pacientes)

    ## Obtengo una lista de todos los IDs de los pacientes para los cuales yo tengo información
    ids_existentes_info = np.array(datos_pacientes['sampleid'])

    ## ---------------------------------------- EXISTENCIA DE DATOS ----------------------------------------

    ## Obtengo una lista con todos los archivos de pacientes para los cuales existe el dataset
    files = os.listdir(ruta_dataset)

    ## Genero una lista en la cual voy a almacenar los IDs de todos aquellos pacientes cuyo registro exista en la base de datos
    ids_existentes_db = []

    ## Itero para cada uno de los registros existentes y me voy guardando el ID
    for file in files:

        ## Obtengo el ID del paciente correspondiente y lo guardo en la lista
        ids_existentes_db.append(int(file[1:]))

    ## Hago la transformación a un vector de tipo numpy
    ids_existentes_db = np.array(ids_existentes_db)

    ## ------------------------------------- CRUCE Y SELECCIÓN DE DATOS ------------------------------------

    ## Hago la intersección entre los IDs de los pacientes para los cuales hay datos en el dataset 
    ## y de los cuales tengo información
    ids_existentes = np.intersect1d(ids_existentes_info, ids_existentes_db)

    ## De todos los pacientes existentes en el dataframe, selecciono únicamente aquellos cuya ID se encuentre en la lista anterior
    ## De éste modo yo obtengo como resultado un dataframe conteniendo únicamente los datos de aquellos pacientes para los cuales existen registros de marcha
    pacientes = datos_pacientes[datos_pacientes["sampleid"].isin(ids_existentes)]

    ## Retorno el dataframe con la información filtrada de los pacientes
    return pacientes, ids_existentes

## Construyo una función la cual me permita obtener la cantidad total de registros relevados por la empresa
def CantidadRegistros():

    ## Construyo un diccionario donde:
    ## Las claves son los nombres de las actividades
    ## Los valores son una tupla donde el primer elemento es el ID de la actividad y el segundo elemento la cantidad de registros de dicha actividad
    actividades = {'Sentado': ['1', 0], 'Parado': ['2', 0], 'Caminando': ['3', 0], 'Escalera': ['4', 0]}

    ## Hago la lectura de los datos generales de los pacientes
    pacientes, ids_existentes = LecturaDatosPacientes()

    ## Itero para cada uno de los identificadores de los pacientes
    for id_persona in ids_existentes:
        
        ## Itero para cada una de las cuatro actividades que tengo detectada
        for actividad in list(actividades.keys()):

            ## Coloco un bloque try en caso de que ocurra algún error de procesamiento
            try:

                ## Ruta del archivo
                ruta = ruta_dataset + "/S{}/{}S{}.csv".format(id_persona, actividades[actividad][0], id_persona)

                ## Selecciono las columnas deseadas (levantará la excepción si no existe tal registro)
                data = pd.read_csv(ruta)

                ## Si el registro efectivamente existe y no da error entonces aumento en una unidad el contador de cantidad de registros
                actividades[actividad][1] += 1

            ## Si hay un error de procesamiento
            except:

                ## Que siga a la siguiente muestra
                continue
    
    ## Retorno el diccionario con las actividades y las cantidades de registros
    return actividades