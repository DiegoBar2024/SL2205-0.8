## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import os
import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path().resolve()).replace('\\','/') + '/sereTestLib')
from parameters import *

## ----------------------------------- LECTURA DE DATOS DEL PACIENTE --------------------------------------

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