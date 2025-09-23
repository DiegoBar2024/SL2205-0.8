## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')
from scipy.signal import *
import numpy as np
from sklearn.cluster import KMeans
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
from LecturaDatosPacientes import *
from parameters import *

## ----------------------------------------- CARGADO DE DATOS -------------------------------------------

## Hago la lectura de los datos generales de los pacientes presentes en la base de datos
pacientes, ids_existentes = LecturaDatosPacientes()

## Inicializo una lista en donde voy a agregar todas las muestras sin separar por pacientes
muestras_juntas = np.zeros((1, 128 * 6 * 800))

## Itero para cada uno de los identificadores de los pacientes
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Especifico la ruta de donde voy a leer los escalogramas
        ruta_lectura = ruta_escalogramas + "/S{}/".format(id_persona)

        ## Obtengo todos los archivos presentes en la ruta anterior
        ## Recuerdo que cada uno de los archivos se corresponde con un escalograma (tensor tridimensional)
        archivos = [archivo for archivo in os.listdir(ruta_lectura) if archivo.endswith("npz")]

        ## Construyo una matriz donde las filas son las observaciones y las columnas los features
        datos_entrada = np.zeros((len(archivos), 6 * 128 * 800))

        ## Itero para todos los archivos presentes en el directorio
        for i in range (len(archivos)):

            ## Abro el archivo con la extensión .npz donde se encuentra el escalograma
            archivo_escalograma = np.load(ruta_lectura + archivos[i])

            ## Almaceno el escalograma correspondiente en la variable <<escalograma>>
            escalograma = archivo_escalograma['X']

            ## Hago un flatten para crear un vector en base al escalograma
            vector_escalograma = np.ndarray.flatten(escalograma)

            ## Agrego el vector conteniendo el escalograma a la matriz correspondiente
            datos_entrada[i, :] = vector_escalograma

        ## Agrego la matriz de muestras al vector
        muestras_juntas = np.concatenate((muestras_juntas, datos_entrada), axis = 0)

        ## Imprimo el número de paciente que fue procesado
        print("ID del paciente procesado: {}".format(id_persona))
    
    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue

## Elimino la primera fila del array de muestras juntas (dummy vector)
muestras_juntas = muestras_juntas[1:,:]

## Guardo todas las muestras juntas en un único archivo
np.savez_compressed(ruta_juntas_sin_comprimir + "/MuestrasJuntas", X = muestras_juntas)