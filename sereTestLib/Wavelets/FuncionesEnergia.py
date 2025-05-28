## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
from scipy import fftpack
import json
from matplotlib import pyplot as plt
from EnergiaDWT import EnergiaDWT
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')

from LecturaDatos import *
from Segmentacion import *
from ContactosIniciales import *
from ContactosTerminales import *

## --------------------------------------- CÁLCULO DE ENERGÍA TOTAL ------------------------------------

def EnergiaTotal(acel, gyro, id_persona):

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

    ## Retorno el diccionario con las energías totales asociadas al paciente
    return energias_totales

## ------------------------------------ CÁLCULO DE ENERGÍA POR SEGMENTO --------------------------------

def EnergiaPorSegmento(acel, gyro, pasos):

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
    
    ## Retorno la matriz que contiene las energías correspondientes a cada canal
    return matriz_energias_canales

## --------------------------------------- CÁLCULO DE ESTADÍSTICAS -------------------------------------

def EstadisticasEnergia(matriz_energias_canales):

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
    
    ## Retorno la matriz con las estadísticas de la energía de cada canal
    return matriz_estadisticos_canales

## Ejecución principal del programa
if __name__== '__main__':

    ## Especifico el ID de la persona para la cual voy a procesar los datos
    id_persona = 204

    ## Hago la lectura de los datos del registro de marcha del paciente
    data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona, lectura_datos_propios = False)

    ## Cálculo de contactos iniciales
    contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = False)

    ## Cálculo de contactos terminales
    contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False)

    ## Hago la segmentación de la marcha
    pasos, duraciones_pasos = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

    ## Hago el cálculo de las energías totales por segmento
    ## La i-ésima fila va a hacer referencia al i-ésimo canal (AC_x, AC_y, AC_z, GY_x, GY_y, GY_z)
    ## La j-ésima columna va a hacer referencia al j-ésimo segmento del registro de marcha
    matriz_energias_canales = EnergiaPorSegmento(acel, gyro, pasos)

    ## Me quedo con las energías correspondientes al canal del giroscopio Y que tiene el eje vertical
    energias_GY_y = matriz_energias_canales[4, :]

    ## Obtengo un vector que tenga los pasos numerados
    pasos_numerados = np.arange(0, len(pasos) - 1, 1)

    ## Hago una gráfica de dispersión observando la energía del giroscopio Y por cada paso
    plt.scatter(pasos_numerados, energias_GY_y)
    plt.show()

    ## Inicializo una matriz en donde me voy a guardar la energía en las subbandas de cada segmento de la señal del giroscopio Y
    ## La i-ésima fila se va a corresponder con el i-ésimo segmento
    ## La j-ésima columna se va a corresponder con la j-ésima subbanda
    energias_subbandas_GY_y = np.zeros((len(pasos) - 1, 9))

    ## Itero para cada uno de los segmentos
    for i in range (1, len(pasos) - 1):

        ## Aislo el segmento de la señal del giroscopio Y en el segmento correspondiente
        gyroY_seg = gyro[:,1][(pasos[i]['IC'][0]) : (pasos[i]['IC'][1])]

        ## Hago la descomposición de energía en subbandas de dicha señal del giroscopio usando DWT
        subbanda_GY_y = EnergiaDWT(gyroY_seg, periodoMuestreo)

        ## Agrego como fila la distribución de energía en la subbanda de la matriz correspondiente
        energias_subbandas_GY_y[i, :] = subbanda_GY_y