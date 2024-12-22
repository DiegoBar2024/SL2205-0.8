import matplotlib.pyplot as plt
import scaleogram as scg
import seaborn as sns
import numpy as np
import pywt
import math
from scipy.integrate import cumulative_trapezoid

def EnergiaSeñalTiempo(señal):
    """
    Función que me calcula la energía total de la señal de entrada en el dominio temporal
    """

    ## Creo una variable donde guardo la energía calculada
    energia = 0

    ## Itero para cada una de las muestras de la señal de entrada
    for muestra in señal:

        ## Actualizo el valor de la energía sumando el cuadrado de la muestra
        energia += np.abs(muestra) ** 2

    ## Retorno el valor de la energía
    return energia

def EnergiaCWT(CWT, escalas):
    """
    Función que calcula la energía total de los coeficientes de la CWT de una señal
    """

    ## Creo una variable donde guardo la energía de la CWT
    energia_cwt = 0

    ## Itero para cada una de las escalas de la CWT las cuales serán las filas de la matriz de coeficientes
    for i in range (len(escalas)):

        ## Itero para cada uno de los valores temporales los cuales serán las columnas de la matriz de coeficientes
        for j in range (CWT.shape[1]):

            ## Actualizo la variable donde guardo la energía de la wavelet
            energia_cwt += (np.abs(CWT[i,j]) / escalas[i]) ** 2
    
    ## Imprimo el valor de la energía de la CWT
    return energia_cwt

def ConstanteAdmisibilidad(f_central, sigma):
    """
    Función que calcula la constante de admisibilidad de una Morlet Compleja con parámetros dados como entrada
    """

    ## Genero un vector de frecuencias arbitrariamente largo
    w = np.arange(0.00000005, 1000, 0.01)

    ## Genero un vector con el valor w0
    w0 = 2 * math.pi * f_central * np.ones(w.shape)

    ## Escribo el integrando en base al análsis teórico
    integrando = (math.e ** ((- sigma * (w - w0) ** 2) / 2)) / w

    ## Aplico la regla trapezoidal para integrar
    Cg = cumulative_trapezoid(integrando, w, initial=0)

    ## Retorno la constante de admisibilidad
    return Cg

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1500.0

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = 2 * np.cos(100.0 * 2.0 * np.pi * x)

## Escalas de las wavelets
escalas = np.arange(0.5, 400, 0.5)

## Ploteo de Escalograma
coef, scales_freq = pywt.cwt(data = y, scales = escalas, wavelet = 'cmor1.5-1', sampling_period = T)

data = np.abs(coef) ** 2
cmap = plt.get_cmap('jet', 256)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
t = np.arange(coef.shape[1]) * T

ax.pcolormesh(t, scales_freq, data, cmap=cmap, vmin=data.min(), vmax=data.max(), shading='auto')

plt.show()

# (cA, cD) = pywt.dwt(data = y, wavelet = 'cmor1.5-1')

print(pywt.wavelist(kind='discrete'))