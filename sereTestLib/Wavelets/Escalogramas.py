## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import tftb
from Augmentado import *
from scipy.signal import *
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from SegmentacionM1 import *

## --------------------------------------------- PARÁMETROS --------------------------------------------

## Escalas de las wavelets a utilizar
## Se recuerda que la frecuencia me queda f = f_muestreo / escala
## Ésto me implica que a escalas más pequeñas tengo frecuencias más grandes
escalas = np.arange(10, 200, 1)

## Creo una variable la cual almacene el ancho de banda de la wavelet
ancho_banda = 1.5

## Tipo de wavelet a utilizar. Wavelet de Morlet Compleja
## Parámetro B (Ancho de banda): 1.5 Hz (ajustable)
## Parámetro C (Frecuencia Central): 1 Hz
wavelet = 'cmor{}-1'.format(ancho_banda)

## ------------------------------------- DIRECTORIOS Y NOMBRES BASE ------------------------------------

## Construyo el directorio en donde se van a guardar los escalogramas
directorio_escalogramas = 'C:/Yo/Tesis/sereData/sereData/Dataset/wavelet_cmor/train/S{}/'.format(id_persona)

## Construyo el directorio en donde se van a guardar los datos preprocesados
directorio_preprocesados = 'C:/Yo/Tesis/sereData/sereData/Dataset/preprocessed_data/train/S{}/'.format(id_persona)

## Creo el nombre base raíz que voy a usar para guardar los escalogramas
nombre_base_segmento = "3S{}s".format(id_persona)

## ------------------------------------------- SEGMENTACIÓN --------------------------------------------

## Creo una matriz donde se van a guardar los segmentos de las señales de acelerómetros y giroscopios según los pasos detectados
## La indexación de <<matriz_segmentada>> en [Dimension1, Dimension2, Dimension3] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica el instante temporal
matriz_segmentada = np.zeros(len(pasos), dtype = object)

## Creo un vector en donde voy guardando las extensiones en muestras correspondientes a cada paso
extensiones_pasos = []

## Especifico la cantidad de pasos que quiero representar en mis escalogramas
cantidad_pasos = 3

## Ahora itero para cada uno de los pasos que tengo segmentados
for i in range (1, len(pasos) - (cantidad_pasos - 1)):

    ## Le doy a la extensión un valor predefinido de 400 muestras (aproximadamente 2 segundos)
    extension = 400

    ## En caso de que la extensión sea mayor al tiempo que tengo en el limite izquierdo, lo seteo en el limite izquierdo
    if extension > pasos[i]['IC'][0]:

        ## Actualizo el valor de la extensión a realizar
        extension = pasos[i]['IC'][0]

    ## En caso de que la extensión sea mayor al tiempo que tengo en el limite derecho, lo seteo en el limite derecho
    elif extension > cant_muestras - pasos[i]['IC'][1]:

        ## Actualizo el valor de la extensión a realizar
        extension = pasos[i]['IC'][1]

    ## Filas de <<matriz>>: Son cada uno de los 6 canales que tengo. Es decir señales de acelerómetros y giroscopios
    ## Columnas de <<matriz>>: Son cada uno de los instantes temporales que componen el paso detectado entre dos ICs
    ## Para mitigar los efectos de borde se le agrega una extensión en ambos lados del intervalo [IC[i] - e, IC[i+1] + e]
    matriz = np.array([ acel[:,0][(pasos[i]['IC'][0] - extension) : (pasos[i + (cantidad_pasos - 1)]['IC'][1] + extension)],
                        acel[:,1][(pasos[i]['IC'][0] - extension) : (pasos[i + (cantidad_pasos - 1)]['IC'][1] + extension)],
                        acel[:,2][(pasos[i]['IC'][0] - extension) : (pasos[i + (cantidad_pasos - 1)]['IC'][1] + extension)],
                        gyro[:,0][(pasos[i]['IC'][0] - extension) : (pasos[i + (cantidad_pasos - 1)]['IC'][1] + extension)],
                        gyro[:,1][(pasos[i]['IC'][0] - extension) : (pasos[i + (cantidad_pasos - 1)]['IC'][1] + extension)],
                        gyro[:,2][(pasos[i]['IC'][0] - extension) : (pasos[i + (cantidad_pasos - 1)]['IC'][1] + extension)] ])

    ## Agrego la matriz al tensor de datos segmentados
    matriz_segmentada[i] = matriz

    ## Guardo el valor de la extensión para el i-ésimo paso
    ## Ésto lo hago para los primeros pasos en donde la extensión puede no ser completa
    extensiones_pasos.append(extension)

## ------------------------------------------- ESCALOGRAMAS --------------------------------------------

## Creo una matriz en la cual para cada paso asocio un tensor tridimensional con las transformadas de wavelet
## La indexación de <<matriz_escalogramas>> en [Dimension1, Dimension2, Dimension3, Dimension4] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica la escala
##  iv)  Dimension4: Indica el instante temporal
matriz_escalogramas = np.zeros(len(pasos), dtype = object)

## En la entrada i-esima entrada habrá un diccionario que contenga la matriz de
## escalogramas del segmento 'i' y el nombre base de ese segmento
escalogramas_segmentos = []

## Itero para cada uno de los segmentos que tengo
for i in range (1, len(pasos) - (cantidad_pasos - 1)):

    ## Imprimo en pantalla el número de iteración que actual
    print("Número de iteración: {} de {}".format(i, len(pasos) - (cantidad_pasos - 1) - 1))

    ## Transformada de Wavelet de la aceleración en el eje x
    coef1, scales_freq = pywt.cwt(data = matriz_segmentada[i][0], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la aceleración en el eje y
    coef2, scales_freq = pywt.cwt(data = matriz_segmentada[i][1], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la aceleración en el eje z
    coef3, scales_freq = pywt.cwt(data = matriz_segmentada[i][2], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la señal de giroscopio respecto del eje x
    coef4, scales_freq = pywt.cwt(data = matriz_segmentada[i][3], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la señal de giroscopio respecto del eje y
    coef5, scales_freq = pywt.cwt(data = matriz_segmentada[i][4], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la señal de giroscopio respecto del eje z
    coef6, scales_freq = pywt.cwt(data = matriz_segmentada[i][5], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Agrego al tensor de escalogramas una pila con los 6 coeficientes
    matriz_escalogramas[i] = np.array([     coef1[:,extensiones_pasos[i - 1]: coef1.shape[1] - extensiones_pasos[i - 1]],
                                            coef2[:,extensiones_pasos[i - 1]: coef2.shape[1] - extensiones_pasos[i - 1]],
                                            coef3[:,extensiones_pasos[i - 1]: coef3.shape[1] - extensiones_pasos[i - 1]],
                                            coef4[:,extensiones_pasos[i - 1]: coef4.shape[1] - extensiones_pasos[i - 1]],
                                            coef5[:,extensiones_pasos[i - 1]: coef5.shape[1] - extensiones_pasos[i - 1]],
                                            coef6[:,extensiones_pasos[i - 1]: coef6.shape[1] - extensiones_pasos[i - 1]]     ])

    ## Hago el remuestreo de la señal temporal para quue el tensor me quede de dimensión fija
    ## En principio le digo que la cantidad objetivo de muestras se calcule como:
    ## el producto entre la cantidad de pasos por ventana y la cantidad promedio de muestras por paso (redondeado a entero)
    matriz_escalogramas[i] = resample(matriz_escalogramas[i], cantidad_pasos * int(muestras_paso), axis = 2)

    ## Obtención del tamaño del tensor tridimensional correspondiente al i-ésimo segmento
    ## Se recuerda que  <<matriz_escalogramas[i]>> tiene la siguiente forma: (c, f, t) para un segmento individual
    ## c: Especifica la cantidad de canales que tengo (en nuestro caso 6 - 3 aceleraciones + 3 giroscopios)
    ## f: Especifica la cantidad de valores de frecuencias que tengo
    ## t: Especifica la cantidad de instantes temporales que tengo
    (c, f, t) = matriz_escalogramas[i].shape
    print((c, f, t))

    ## Especifico el proceso de guardado del escalograma en la lista escalogramas_segmentos
    ## Genero el nombre del archivo de salida
    ## Me queda definir <<directorio_scalogramas>> y <<nombre_base_segmento>>
    archivo_salida = '%s%s' % (directorio_escalogramas, "{}{}".format(nombre_base_segmento, str(i - 1)))

    ## Especifico el nombre de archivo de salida de los datos preprocesados
    archivo_preprocesado = '%s%s' % (directorio_preprocesados, "{}{}".format(nombre_base_segmento, i - 1))

    ## Hago el guardado de los escalogramas
    for j, dato in zip(range(6), ['ACx', 'ACy', 'ACz', 'GYx', 'GYy', 'GYz']):

        ## Hago el guardado de los escalogramas de cada canal
        np.savez_compressed(archivo_salida + str(dato) + '.npz', y = 0, X = matriz_escalogramas[i][j,:,:])

        ## Hago el guardado de los escalogramas preprocesados de cada canal
        np.savez_compressed(archivo_preprocesado + "_00" + j + '.npz', y = 0, X = matriz_escalogramas[i][j,:,:])

    ## Agrego el escalograma y su nombre base como un diccionario a la lista de escalogramas
    escalogramas_segmentos.append({'escalograma': matriz_escalogramas, 'nombre_base_segmento': nombre_base_segmento})

    # ## Graficación del escalograma en el i-ésimo segmento (anterposterior)
    # data = np.abs(coef3[:,extensiones_pasos[i]: coef3.shape[1] - extensiones_pasos[i]])
    # cmap = plt.get_cmap('jet', 256)
    # fig = plt.figure(figsize = (5,5))
    # ax = fig.add_subplot(111)
    # t = np.arange(coef3[:,extensiones_pasos[i]: coef3.shape[1] - extensiones_pasos[i]].shape[1]) * periodoMuestreo
    # ax.pcolormesh(t, scales_freq, data, cmap=cmap, vmin=data.min(), vmax=data.max(), shading='auto')
    # plt.xlabel("Tiempo (s)")
    # plt.ylabel("Frecuencia (Hz)")
    # plt.title("$|CWT(t,f)|$")
    # plt.show()

    # ## Hago remuestreo de la señal de aceleración anteroposterior
    # acc_ap_remuestreo = resample(acel[:,2][pasos[i]['IC'][0] : pasos[i + (cantidad_pasos - 1)]['IC'][1]], cantidad_pasos * int(muestras_paso))

    # ## Graficación de la señal en el tiempo (anteroposterior) comparando con la señal remuestreada
    # plt.plot(acel[:,2][pasos[i]['IC'][0] : pasos[i + (cantidad_pasos - 1)]['IC'][1]])
    # plt.plot(acc_ap_remuestreo)
    # plt.show()

## ------------------------------------------- AUGMENTADO ----------------------------------------------

## El parámetro <<directorio_scalogramas_train>> va a tener la ruta donde están los escalogramas que voy a usar para procesar
## Para el caso de MI COMPUTADORA <<directorio_scalogramas_train>> = 'C:\Yo\Tesis\sereData\sereData\Dataset\wavelet_cmor\train'
## El parámetro <<dir_preprocessed_data_train>> me va a almacenar los escalogramas de salida luego del preprocesado
## Para el caso de MI COMPUTADORA <<dir_preprocessed_data_train>> = 'C:\Yo\Tesis\sereData\sereData\Dataset\preprocessed_data\train'
## El parámetro <<directorio_muestra>> me va a contener la carpeta correspondiente al paciente analizado y es el que se arma en las líneas de arriba
## Por ejemplo, si se está analizando una muestra corta (<<long_sample>> = False) del paciente de ID 215 (<<sample_ID>> = 215)
## El parámetro <<act_ae>> es una lista que contiene las actividades que yo quiero procesar. Se importa del fichero <<parameters.py>> y tiene como valor <<act_ae>> = ['Caminando']
## LA SALIDA DE ESTA FUNCIÓN ES LA QUE ENTRA AL MÓDULO DE INTELIGENCIA ARTIFICIAL    
create_or_augment_scalograms(directorio_scalogramas_train + directorio_muestra, dir_preprocessed_data_train + directorio_muestra, static_window_secs = static_window_secs, actividades = act_ae, movil_window_secs = window_secs, overlap_secs = overlap_secs, fs = fs, escalado = escalado)