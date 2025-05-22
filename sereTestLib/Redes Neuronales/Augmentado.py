## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import random
import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing import image
from matplotlib import pyplot as plt
import os
from PIL import Image as im 

## ------------------------------------ FUNCIONES DE INYECCIÓN DE RUIDO --------------------------------

## Función que me agrega ruido aleatorio a las imágenes
def add_noise(img):

    ## Defino la variabilidad del ruido (parámetro que se usa para obtener la desviación estándar del ruido blanco)
    VARIABILITY = 50

    ## Especifico cuál va a ser la desviación estándar del ruido
    deviation = VARIABILITY * random.random()

    ## Construyo un array bidimensional de ruido blanco gaussiano de media nula y desviación estándar
    noise = np.random.normal(0, deviation, img.shape)

    ## Hago la adición del ruido blanco a la imagen
    img += noise

    ## Limito los valores del array para que estén entre 0 y 255 (intensidades de píxeles de gris)
    np.clip(img, 0., 255.)

    ## Retorno la imagen ruidosa
    return img

## Función que me genera imágenes distorsionadas a partir de una imagen normal
def InyeccionRuido(imagen, generador_img, cant_imagenes = 4):

    ## Inicializo la lista de imágenes colocando mi imagen original como primer elemento
    images = [imagen]
    
    ## Hago el pasaje de la imagen de entrada a un array bidimensional cuyos valores son los píxeles
    ## En realidad me queda en la forma de un tensor tridimensional pero que tiene únicamente un canal
    img_arr = image.img_to_array(imagen)

    ## Agrego una dimensión adicional al tensor de la imagen como primera coordenada
    img_arr = img_arr.reshape((1,) + img_arr.shape)

    ## Itero para cada una de las imágenes distorsionadas
    ## Cada imagen distorsionada va a estar contaminada por un ruido aleatorio gaussiano de media nula
    for batch in generador_img.flow(img_arr, batch_size = 1):

        ## Agrego la imagen distorsionada a la lista de imágenes
        images.append(image.array_to_img(batch[0]))

        ## En caso de que la cantidad de imágenes en la lista sea mayor o igual a 4
        if len(images) >= cant_imagenes:

            ## Termino el bucle. Ésto me limita la cantidad de imágenes distorsionadas que estoy generando
            break
    
    ## Retorno la lista de imágenes distorsionadas
    return images

## ----------------------------------- GENERACIÓN DE DATASET RUIDOSO -----------------------------------

## Defino el generador de imágenes especificando como función de preprocesamiento la adición de ruido
generador_img = ImageDataGenerator(preprocessing_function = add_noise)

## Especifico el ID de la persona para el que voy a guardar las imagenes
id_persona = 299

## La idea es poder guardar todos los escalogramas generados como imágenes con extensión .png en una carpeta aparte
## Especifico la ruta de donde voy a leer los escalogramas
ruta_lectura = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo_gp/S{}/".format(id_persona)

## Especifico la ruta donde voy a guardar los escalogramas como imágenes
ruta_guardado = "C:/Yo/Tesis/sereData/sereData/Escalogramas_gp_aug/S{}/".format(id_persona)

## Especifico la ruta donde voy a guardar los escalogramas augmentados
ruta_guardado_aug = "C:/Yo/Tesis/sereData/sereData/Dataset/escalogramas_nuevo_gp_aug/S{}/".format(id_persona)

## En caso de que el directorio no exista
if not os.path.exists(ruta_guardado):

    ## Creo el directorio correspondiente
    os.makedirs(ruta_guardado)

## En caso de que el directorio no exista
if not os.path.exists(ruta_guardado_aug):

    ## Creo el directorio correspondiente
    os.makedirs(ruta_guardado_aug)

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

        ## Guardo la imagen distorsionada como PNG
        imagenes_AC_x[j].save(ruta_guardado + "3S{}s{}_ACx{}.png".format(id_persona, i, j))

        ## Guardo la imagen distorsionada como un tensor tridimensional para luego poder entrenar la red
        np.savez_compressed(ruta_guardado_aug + "3S{}sACx{}{}".format(id_persona, i, j), X = np.asarray(imagenes_AC_x[j]))

    ## Genero las imágenes distorsionadas
    imagenes_AC_y = InyeccionRuido(imagen_AC_y, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_AC_y)):

        ## Guardo la imagen distorsionada
        imagenes_AC_y[j].save(ruta_guardado + "3S{}s{}_ACy{}.png".format(id_persona, i, j))

        ## Guardo la imagen distorsionada como un tensor tridimensional para luego poder entrenar la red
        np.savez_compressed(ruta_guardado_aug + "3S{}sACy{}{}".format(id_persona, i, j), X = np.asarray(imagenes_AC_y[j]))

    ## Genero las imágenes distorsionadas
    imagenes_AC_z = InyeccionRuido(imagen_AC_z, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_AC_z)):

        ## Guardo la imagen distorsionada
        imagenes_AC_z[j].save(ruta_guardado + "3S{}s{}_ACz{}.png".format(id_persona, i, j))

        ## Guardo la imagen distorsionada como un tensor tridimensional para luego poder entrenar la red
        np.savez_compressed(ruta_guardado_aug + "3S{}sACz{}{}".format(id_persona, i, j), X = np.asarray(imagenes_AC_z[j]))

    ## Genero las imágenes distorsionadas
    imagenes_GY_x = InyeccionRuido(imagen_GY_x, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_GY_x)):

        ## Guardo la imagen distorsionada
        imagenes_GY_x[j].save(ruta_guardado + "3S{}s{}_GYx{}.png".format(id_persona, i, j))

        ## Guardo la imagen distorsionada como un tensor tridimensional para luego poder entrenar la red
        np.savez_compressed(ruta_guardado_aug + "3S{}sGYx{}{}".format(id_persona, i, j), X = np.asarray(imagenes_GY_x[j]))

    ## Genero las imágenes distorsionadas
    imagenes_GY_y = InyeccionRuido(imagen_GY_y, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_GY_y)):

        ## Guardo la imagen distorsionada
        imagenes_GY_y[j].save(ruta_guardado + "3S{}s{}_GYy{}.png".format(id_persona, i, j))

        ## Guardo la imagen distorsionada como un tensor tridimensional para luego poder entrenar la red
        np.savez_compressed(ruta_guardado_aug + "3S{}sGYy{}{}".format(id_persona, i, j), X = np.asarray(imagenes_GY_y[j]))

    ## Genero las imágenes distorsionadas
    imagenes_GY_z = InyeccionRuido(imagen_GY_z, generador_img)

    ## Itero para cada una de las imágenes distorsionadas
    for j in range (len(imagenes_GY_z)):

        ## Guardo la imagen distorsionada
        imagenes_GY_z[j].save(ruta_guardado + "3S{}s{}_GYz{}.png".format(id_persona, i, j))

        ## Guardo la imagen distorsionada como un tensor tridimensional para luego poder entrenar la red
        np.savez_compressed(ruta_guardado_aug + "3S{}sGYz{}{}".format(id_persona, i, j), X = np.asarray(imagenes_GY_z[j]))