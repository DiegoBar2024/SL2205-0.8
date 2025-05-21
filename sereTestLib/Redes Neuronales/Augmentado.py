## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import random
import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing import image
from matplotlib import pyplot as plt
import os
from PIL import Image as im 

## ---------------------------------------- INYECCIÓN DE RUIDO -----------------------------------------

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def InyeccionRuido(imagen, generador_img):

    ## Genero las imágenes distorsionadas
    images = [imagen]
    img_arr = image.img_to_array(imagen)
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    for batch in generador_img.flow(img_arr, batch_size = 1):
        images.append( image.array_to_img(batch[0]) )
        if len(images) >= 4:
            break
    
    ## Retorno las imágenes generadas
    return images

    # ## Graficación de las imágenes distorsionadas
    # f, xyarr = plt.subplots(2,2)
    # xyarr[0,0].imshow(images[0])
    # xyarr[0,1].imshow(images[1])
    # xyarr[1,0].imshow(images[2])
    # xyarr[1,1].imshow(images[3])
    # plt.show()

## ----------------------------------- GENERACIÓN DE DATASET RUIDOSO -----------------------------------

## Defino el generador de imágenes especificando como función de preprocesamiento la adición de ruido
generador_img = ImageDataGenerator(preprocessing_function = add_noise)

## Especifico el ID de la persona para el que voy a guardar las imagenes
id_persona = 299

## La idea es poder guardar todos los escalogramas generados como imágenes con extensión .png en una carpeta aparte
## Especifico la ruta de donde voy a leer los escalogramas
ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo_gp/S{}/".format(id_persona)

## Especifico la ruta de donde voy a guardar los escalogramas
ruta_guardado = "C:/Yo/Tesis/sereData/sereData/Escalogramas_gp_aug/S{}/".format(id_persona)

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

    ## Genero las imágenes distorsionadas
    imagenes_AC_x = InyeccionRuido(imagen_AC_x, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_AC_x)):

        ## Guardo la imagen distorsionada
        imagenes_AC_x[j].save(ruta_guardado + "3S{}s{}_ACx{}.png".format(id_persona, i, j))
    
    ## Genero las imágenes distorsionadas
    imagenes_AC_y = InyeccionRuido(imagen_AC_y, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_AC_y)):

        ## Guardo la imagen distorsionada
        imagenes_AC_y[j].save(ruta_guardado + "3S{}s{}_ACy{}.png".format(id_persona, i, j))

    ## Genero las imágenes distorsionadas
    imagenes_AC_z = InyeccionRuido(imagen_AC_z, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_AC_z)):

        ## Guardo la imagen distorsionada
        imagenes_AC_z[j].save(ruta_guardado + "3S{}s{}_ACz{}.png".format(id_persona, i, j))

    ## Genero las imágenes distorsionadas
    imagenes_GY_x = InyeccionRuido(imagen_GY_x, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_GY_x)):

        ## Guardo la imagen distorsionada
        imagenes_GY_x[j].save(ruta_guardado + "3S{}s{}_GYx{}.png".format(id_persona, i, j))

    ## Genero las imágenes distorsionadas
    imagenes_GY_y = InyeccionRuido(imagen_GY_y, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_GY_y)):

        ## Guardo la imagen distorsionada
        imagenes_GY_y[j].save(ruta_guardado + "3S{}s{}_GYy{}.png".format(id_persona, i, j))
    
    ## Genero las imágenes distorsionadas
    imagenes_GY_z = InyeccionRuido(imagen_GY_z, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_GY_z)):

        ## Guardo la imagen distorsionada
        imagenes_GY_z[j].save(ruta_guardado + "3S{}s{}_GYz{}.png".format(id_persona, i, j))