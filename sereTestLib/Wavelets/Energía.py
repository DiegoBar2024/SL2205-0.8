## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
from scipy import fftpack
import json
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from SegmentacionM1 import *

## --------------------------------------- CÁLCULO DE ENERGÍA TOTAL ------------------------------------

## Obtengo la energía total correspondiente a la señal de aceleración en el eje x
energia_ac_x = np.sum(np.square(acel[:,0]))

## Obtengo la energía total correspondiente a la señal de aceleración en el eje y
energia_ac_y = np.sum(np.square(acel[:,1]))

## Obtengo la energía total correspondiente a la señal de aceleración en el eje z
energia_ac_z = np.sum(np.square(acel[:,2]))

## Obtengo la energía total correspondiente a la señal de giroscopio en el eje x
energia_gy_x = np.sum(np.square(gyro[:,0]))

## Obtengo la energía total correspondiente a la señal de giroscopio en el eje y
energia_gy_y = np.sum(np.square(gyro[:,1]))

## Obtengo la energía total correspondiente a la señal de giroscopio en el eje z
energia_gy_z = np.sum(np.square(gyro[:,2]))

## Asocio el número identificador del paciente con una tupla conteniendo las energías totales de su registro de marcha
energias_totales = {id_persona : (energia_ac_x, energia_ac_y, energia_ac_z, energia_gy_x, energia_gy_y, energia_gy_z)}

## ------------------------------------- ESCRITURA EN BASE DE DATOS ------------------------------------

## Hago la lectura del archivo JSON de energías previamente existente
with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias.json", 'r') as openfile:

    # Cargo el diccionario el cual va a ser un objeto JSON
    dicc_energias = json.load(openfile)

## Agrego en el diccionario los datos de energías del paciente actual
dicc_energias[id_persona] = energias_totales[id_persona]

## Especifico la ruta del archivo JSON sobre la cual voy a reescribir el diccionario de energías
with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias.json", "w") as outfile:

    ## Escribo el diccionario actualizado
    json.dump(dicc_energias, outfile)

## ------------------------------------ CÁLCULO DE ENERGÍA POR SEGMENTO --------------------------------

## Creo una matriz donde se van a guardar los segmentos de las señales de acelerómetros y giroscopios según los pasos detectados
## La indexación de <<matriz_segmentada>> en [Dimension1, Dimension2, Dimension3] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica el instante temporal
matriz_segmentada = np.zeros(len(pasos), dtype = object)

## Especifico la cantidad de pasos que quiero representar en mis escalogramas
cantidad_pasos = 1

## Calculo la cantidad de segmentos que se van a tener en base a las cantidades anteriores
cantidad_segmentos = len(pasos) - cantidad_pasos

## Creo una matriz que me guarde la energía de cada canal en cada segmento
## Las columnas y las filas de la matriz tienen el siguiente significado:
##  FILAS: La i-ésima fila de la matriz va a hacer referencia al i-ésimo canal
##  COLUMNAS: La j-ésima columna de la matriz va a hacer referencia al j-ésimo segmento
matriz_energias_canales = np.zeros((6, cantidad_segmentos))

## Ahora itero para cada uno de los pasos que tengo segmentados
for i in range (1, cantidad_segmentos):

    ## Filas de <<matriz>>: Son cada uno de los 6 canales que tengo. Es decir señales de acelerómetros y giroscopios
    ## Columnas de <<matriz>>: Son cada uno de los instantes temporales que componen el paso detectado entre dos ICs
    ## Para mitigar los efectos de borde se le agrega una extensión en ambos lados del intervalo [IC[i] - e, IC[i+1] + e]
    matriz = np.array([ acel[:,0][(pasos[i]['IC'][0]) : (pasos[i + (cantidad_pasos - 1)]['IC'][1])],
                        acel[:,1][(pasos[i]['IC'][0]) : (pasos[i + (cantidad_pasos - 1)]['IC'][1])],
                        acel[:,2][(pasos[i]['IC'][0]) : (pasos[i + (cantidad_pasos - 1)]['IC'][1])],
                        gyro[:,0][(pasos[i]['IC'][0]) : (pasos[i + (cantidad_pasos - 1)]['IC'][1])],
                        gyro[:,1][(pasos[i]['IC'][0]) : (pasos[i + (cantidad_pasos - 1)]['IC'][1])],
                        gyro[:,2][(pasos[i]['IC'][0]) : (pasos[i + (cantidad_pasos - 1)]['IC'][1])] ])
    
    ## Itero para cada uno de los seis canales que tengo
    for canal in range (matriz.shape[0]):

        ## ---- MÉTODO 1: ENERGÍA DE LA SEÑAL EN EL DOMINIO DEL TIEMPO ----
        ## Creo una variable donde guardo la energía calculada
        energia_tiempo = 0

        ## Itero para cada una de las muestras de la señal de entrada
        for muestra in matriz[canal,:]:

            ## Actualizo el valor de la energía sumando el cuadrado de la muestra
            energia_tiempo += np.abs(muestra) ** 2
        
        ## ---- MÉTODO 2: ENERGÍA DE LA SEÑAL EN EL DOMINIO DE LA FRECUENCIA ----
        ## Calculo el espectro en el semieje derecho de la señal
        espectro = fftpack.rfft(matriz[canal,:])

        ## Calculo la energía total de la señal usando la formula de Parseval en frecuencia
        energia_fourier = (espectro[0] ** 2 + 2 * np.sum(espectro[1:] ** 2)) / len(matriz[canal,:])

        ## Creo una tupla con las energías del segmento en el tiempo y en frecuencia
        ## Hago el cálculo de energía con los dos métodos para comprobar que el resultado esté correcto
        energias_segmento = (energia_tiempo, energia_fourier)

        ## Agrego la energía del segmento a la matriz <<matriz_energías_canales>> en su lugar correspondiente
        ## Uso la energía temporal como la energía total que voy a tomar del segmento
        matriz_energias_canales[canal, i] = energias_segmento[0]

## --------------------------------------- CÁLCULO DE ESTADÍSTICAS -------------------------------------

## Inicializo una variable en donde especifico que quiero hacer un guardado de los datos estadísticos
hacer_guaradado = False

## Creo una matriz que me guarde los estadísticos de energía asociados a cada canal
## Las columnas y las filas de la matriz tienen el siguiente significado:
##  FILAS: La i-ésima fila de la matriz va a hacer referencia al i-ésimo canal
##  COLUMNAS: La primera columna es la energía media, la segunda columna es la mediana, la tercera columna la desviación estándar
matriz_estadisticos_canales = np.zeros((6, 3))

## Itero en cada uno de los canales que tengo
for canal in range (matriz_energias_canales.shape[0]):

    ## Me quedo con el vector de energías correspondientes a dicho canal
    energias_canal = matriz_energias_canales[canal, :]

    ## Hago el cálculo de la energía promedio para ese canal PARA TODOS LOS SEGMENTOS
    energia_media = np.mean(energias_canal)

    ## Hago el cálculo de la mediana en el vector de energías para ese canal PARA TODOS LOS SEGMENTOS
    energia_mediana = np.median(energias_canal)

    ## Hago el cálculo de la desviación estándar de las energías para ese canal PARA TODOS LOS SEGMENTOS
    energia_desv_estandar = np.std(energias_canal)

    ## Agrego los estadísticos correspondientes a la matriz
    matriz_estadisticos_canales[canal, :] = np.array((energia_media, energia_mediana, energia_desv_estandar))