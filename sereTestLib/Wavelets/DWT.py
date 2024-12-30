import matplotlib.pyplot as plt
import scaleogram as scg
import seaborn as sns
import numpy as np
import pywt
import math
from scipy.integrate import cumulative_trapezoid

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1500.0

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = 2 * np.cos(100.0 * 2.0 * np.pi * x) + 2 * np.sin(75.0 * 2.0 * np.pi * x)

## Energía de la señal
## Creo una variable donde guardo la energía calculada
energia_señal = 0

## Itero para cada una de las muestras de la señal de entrada
for muestra in y:

    ## Actualizo el valor de la energía sumando el cuadrado de la muestra
    energia_señal += np.abs(muestra) ** 2

## Conjunto de wavelets discretas disponibles
wavelets_discretas = pywt.wavelist(kind='discrete')

## Genero un diccionario cuyas claves sean los nombres de las wavelets
## Los valores van a ser las tuplas con las descomposiciones en primer nivel
wavelets = {}

## Itero para cada una de las wavelets discretas
for wavelet in wavelets_discretas:

    ## Hago la descomposición en DWT de la señal
    (cA, cD) = pywt.dwt(data = y, wavelet = wavelet, mode = 'periodization')

    ## Agrego la descomposición al diccionario como una lista
    wavelets[wavelet] = [cA, cD]

## Itero para cada una de las wavelets discretas
for wavelet in wavelets_discretas:

    ## Calculo la cantidad total de coeficientes en el nivel 1
    cantidad_coeficientes = len(wavelets[wavelet][0]) + len(wavelets[wavelet][1])

    ## Declaro una variable en donde voy a guardar la energía de la señal
    energia = 0

    ## Itero para cada uno de los coeficientes de aproximación
    for coef in cA:

        ## Calculo el módulo al cuadrado de dicho coeficiente de aproximación
        energia += np.abs(coef) ** 2

    ## Itero para cada uno de los coeficientes de detalle
    for coef in cD:

        ## Calculo el módulo al cuadrado de dicho coeficiente de detalle
        energia += np.abs(coef) ** 2