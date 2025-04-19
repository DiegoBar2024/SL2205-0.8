## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
import pywt
from scipy.signal import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib')

from LecturaDatos import *
from Fourier import *

## ---------------------------------------- CARGADO DEL FICHERO ----------------------------------------

## En caso de que el directorio no exista
if not os.path.exists('data.pkl'):
    
    ## Creo el directorio correspondiente
    os.makedirs('data.pkl')

with open('data.pkl', 'rb') as file:
    res = pickle.load(file)

## ---------------------------------------- CÁLCULO DE ENERGÍA -----------------------------------------

## Me quedo con la aceleración mediolateral
acc_ML = acel[:, 0]

## Hago la graficación del espectro de la señal de aceleración mediolateral
TransformadaFourier(acc_ML, periodoMuestreo, plot = False)

## Descomposición multinivel usando DWT
coefs = pywt.wavedec(data = acc_ML, wavelet = 'dmey', mode = 'periodization', level = 11)

## Seteo la variable donde guardo el valor de la energía en 0
energia_multinivel = 0

## Creo una lista donde asocio los coeficientes con las sub bandas de frecuencia
subbandas = []

## Itero para cada una de las listas de coeficientes
for i in range (1, len(coefs)):

    ## Actualizo el valor de la energía agregando el total de los cuadrados de los coeficientes
    energia_multinivel += np.sum(np.square(coefs[i]))

    ## Especifico el rango de frecuencias correspondiente a la subbanda
    rango = [2 ** ( - (i + 1)) / periodoMuestreo, 2 ** ( - i ) / periodoMuestreo]

    ## Asocio los coeficientes con la subbanda correspondiente y su energia
    subbandas.append((rango, np.sum(np.square(coefs[len(coefs) - i])), coefs[len(coefs) - i]))

## Asigno el rango correspondiente a los coefs de aproximación
rango = [0, 2 ** ( - (len(coefs))) / periodoMuestreo]

## Sumo la energia correspodniente a la aproximación
energia_multinivel += np.sum(np.square(coefs[0]))

## Agrego los coeficientes de aproximación asociandolos con sus bandas de frecuencia y su energía
subbandas.append((rango, np.sum(np.square(coefs[0])), coefs[0]))

## Creo una lista donde guardo los rangos
rangos = []

## Creo una lista donde guardo las energias
energias = []

## Itero para cada una de las bandas correspondientes de la descomposición
for banda in subbandas[::-1]:

    ## Agrego el rango a la lista de energía
    rangos.append("{} - {}".format(round(banda[0][0], 3), round(banda[0][1], 3)))

    ## Recomendación extraída de "A Comprehensive Assessment of Gait Accelerometry Signals in Time, Frequency and Time-Frequency Domains"
    ## En caso de que esté en la banda de interés (en mi caso 0.78 a 1.57 Hz), me quedo con la energía relativa en dicha banda
    if banda[0][1] == 1 / (periodoMuestreo * 2 ** 7):

        ## Me guardo la proporción de energía de dicha banda en una variable
        energia_banda = banda[1] / energia_multinivel

    ## Agrego la energia relativa a la lista de energias
    energias.append(banda[1] / energia_multinivel)

## Hago el gráfico de barras correspondiente
plt.bar(np.array(rangos), np.array(energias))
plt.xticks(rotation = 10)
plt.xlabel("Frecuencias (Hz)")
plt.ylabel("Energía Relativa")
plt.show()