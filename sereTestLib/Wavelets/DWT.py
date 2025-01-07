import matplotlib.pyplot as plt
import scaleogram as scg
import seaborn as sns
import numpy as np
import pywt
from scipy.fft import fft, ifft, fftfreq
from scipy import fftpack
from scipy.integrate import cumulative_trapezoid

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 1500

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = 2 * np.cos(100.0 * 2.0 * np.pi * x) + 2 * np.sin(600 * 2.0 * np.pi * x)

## ENERGÍA DE LA SEÑAL EN EL DOMINIO DEL TIEMPO
## Creo una variable donde guardo la energía calculada
energia_señal = 0

## Itero para cada una de las muestras de la señal de entrada
for muestra in y:

    ## Actualizo el valor de la energía sumando el cuadrado de la muestra
    energia_señal += np.abs(muestra) ** 2

## ENERGÍA DE LA SEÑAL EN EL DOMINIO DE LA FRECUENCIA
## Calculo el espectro en el semieje derecho de la señal y
espectro = fftpack.rfft(y)

## Calculo la energía total de la señal usando la formula de Parseval en frecuencia
energia_fourier = (espectro[0] ** 2 + 2 * np.sum(espectro[1:] ** 2)) / N

## TRANSFORMADA DISCRETA DE WAVELET
## Conjunto de wavelets discretas disponibles
wavelets_discretas = pywt.wavelist(kind = 'discrete')

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

    ## CÁLCULO DE COEFICIENTES
    ## Calculo la cantidad total de coeficientes en el nivel 1
    cantidad_coeficientes = len(wavelets[wavelet][0]) + len(wavelets[wavelet][1])

    ## CÁLCULO DE ENERGIA
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

## Descomposición multinivel usando DWT
coefs = pywt.wavedec(data = y, wavelet = 'db4', mode = 'periodization', level = 4)

energia_multinivel = 0

for coef in coefs:

    energia_multinivel += np.sum(np.square(coef))

print(energia_multinivel)

# ## TRANSFORMADA DE FOURIER DE LA SEÑAL ORIGINAL
# ## Por medio de la función <<fft>> calculo la transformada de Fourier de la señal de entrada
# transformada = fft(y)

# ## Por medio de la función <<fftfreq>> calculo el eje de las frecuencias
# frecuencias = fftfreq(y.shape[0], T)

# ## Hago la gráfica de los coeficientes de la transformada en función de la frecuencia
# ## La gráfica se hace únicamente para frecuencias positivas
# plt.plot(frecuencias[:y.shape[0]//2], (2 / y.shape[0]) * np.abs(transformada[0:y.shape[0]//2]))

# ## Nomenclatura de ejes
# plt.xlabel("Frecuencia (Hz)")
# plt.ylabel("Magnitud")

# ## Despliego la gráfica
# plt.show()

# ## TRANSFORMADA DE FOURIER DE LA APROXIMATION SIGNAL
# ## Por medio de la función <<fft>> calculo la transformada de Fourier de la señal de entrada
# transformada = fft(wavelets['db1'][0])

# ## Por medio de la función <<fftfreq>> calculo el eje de las frecuencias
# frecuencias = fftfreq(wavelets['db1'][0].shape[0], T)

# ## Hago la gráfica de los coeficientes de la transformada en función de la frecuencia
# ## La gráfica se hace únicamente para frecuencias positivas
# plt.plot(frecuencias[:wavelets['db1'][0].shape[0]//2], (2 / wavelets['db1'][0].shape[0]) * np.abs(transformada[0:wavelets['db1'][0].shape[0]//2]))

# ## Nomenclatura de ejes
# plt.xlabel("Frecuencia (Hz)")
# plt.ylabel("Magnitud")

# ## Despliego la gráfica
# plt.show()

# ## TRANSFORMADA DE FOURIER DE LA DETAIL SIGNAL
# ## Por medio de la función <<fft>> calculo la transformada de Fourier de la señal de entrada
# transformada = fft(wavelets['db1'][1])

# ## Por medio de la función <<fftfreq>> calculo el eje de las frecuencias
# frecuencias = fftfreq(wavelets['db1'][1].shape[0], T)

# ## Hago la gráfica de los coeficientes de la transformada en función de la frecuencia
# ## La gráfica se hace únicamente para frecuencias positivas
# plt.plot(frecuencias[:wavelets['db1'][1].shape[0]//2], (2 / wavelets['db1'][1].shape[0]) * np.abs(transformada[0:wavelets['db1'][1].shape[0]//2]))

# ## Nomenclatura de ejes
# plt.xlabel("Frecuencia (Hz)")
# plt.ylabel("Magnitud")

# ## Despliego la gráfica
# plt.show()

# plt.plot(wavelets['db1'][1])
# plt.plot(y)
# plt.show()