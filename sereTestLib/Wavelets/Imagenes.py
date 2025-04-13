## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
import os
from PIL import Image as im 
from matplotlib import pyplot as plt
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

## ------------------------------------ TRANSFORMACIÓN A IMÁGENES --------------------------------------

## Especifico el ID de la persona la cual voy a guardar las imagenes de los escalogramas
id_persona = 114

## La idea es poder guardar todos los escalogramas generados como imágenes con extensión .png en una carpeta aparte
## Especifico la ruta de donde voy a leer los escalogramas
ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo/train/S{}/".format(id_persona)

## Especifico la ruta de donde voy a guardar los escalogramas
ruta_guardado = "C:/Yo/Tesis/sereData/sereData/Escalogramas/S{}/".format(id_persona)

## En caso de que el directorio no exista
if not os.path.exists(ruta_guardado):

    ## Creo el directorio correspondiente
    os.makedirs(ruta_guardado)

## Obtengo todos los archivos presentes en la ruta anterior
## Recuerdo que cada uno de los archivos se corresponde con un escalograma (tensor tridimensional)
archivos = [archivo for archivo in os.listdir(ruta_lectura) if archivo.endswith("npz")]

## Itero para todos los archivos presentes en el directorio
for i in range (len(archivos)):

    ## Abro el archivo con la extensión .npz donde se encuentra el escalograma
    archivo_escalograma = np.load(ruta_lectura + archivos[i])

    ## Almaceno el escalograma correspondiente en la variable <<escalograma>>
    escalograma = archivo_escalograma['X']

    ## Hago la traducción de los escalogramas a imágenes
    imagen_AC_x = im.fromarray(escalograma[0]).convert("L")
    imagen_AC_y = im.fromarray(escalograma[1]).convert("L")
    imagen_AC_z = im.fromarray(escalograma[2]).convert("L")
    imagen_GY_x = im.fromarray(escalograma[3]).convert("L")
    imagen_GY_y = im.fromarray(escalograma[4]).convert("L")
    imagen_GY_z = im.fromarray(escalograma[5]).convert("L")

    ## Hago el guardado de los escalogramas como imagenes en la ruta correspondiente
    imagen_AC_x.save(ruta_guardado + "3S{}s{}_ACx.png".format(id_persona, i))
    imagen_AC_y.save(ruta_guardado + "3S{}s{}_ACy.png".format(id_persona, i))
    imagen_AC_z.save(ruta_guardado + "3S{}s{}_ACz.png".format(id_persona, i))
    imagen_GY_x.save(ruta_guardado + "3S{}s{}_GYx.png".format(id_persona, i))
    imagen_GY_y.save(ruta_guardado + "3S{}s{}_GYy.png".format(id_persona, i))
    imagen_GY_z.save(ruta_guardado + "3S{}s{}_GYz.png".format(id_persona, i))