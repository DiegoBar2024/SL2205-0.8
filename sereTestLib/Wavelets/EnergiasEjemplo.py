import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import fftpack

# Cantidad de muestras
N = 600

# Período de muestreo
T = 1.0 / 200

## Vector de tiempos
x = np.linspace(0.0, N * T, N, endpoint = False)

## Función
y = 2 * np.cos(20 * 2.0 * np.pi * x) + 2 * np.cos(2.25 * 2.0 * np.pi * x) + 20 * np.cos(75 * 2.0 * np.pi * x)

## ENERGÍA DE LA SEÑAL EN EL DOMINIO DEL TIEMPO

## Método rápido
energia_señal_rapido = np.sum(np.square(y))

print("Energía de la señal con metodo rapido: {}".format(energia_señal_rapido))

## Creo una variable donde guardo la energía calculada
energia_señal = 0

## Itero para cada una de las muestras de la señal de entrada
for muestra in y:

    ## Actualizo el valor de la energía sumando el cuadrado de la muestra
    energia_señal += np.abs(muestra) ** 2

print("Energia en el tiempo: {}".format(energia_señal))

## ENERGÍA DE LA SEÑAL EN EL DOMINIO DE LA FRECUENCIA
## Calculo el espectro en el semieje derecho de la señal y
espectro = fftpack.rfft(y)

## Calculo la energía total de la señal usando la formula de Parseval en frecuencia
energia_fourier = (espectro[0] ** 2 + 2 * np.sum(espectro[1:] ** 2)) / N

print("Energia en frecuencia: {}".format(energia_fourier))

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

print("Energía DWT: {}".format(energia))

## Descomposición multinivel usando DWT
coefs = pywt.wavedec(data = y, wavelet = 'db20', mode = 'periodization', level = 10)

## Seteo la variable donde guardo el valor de la energía en 0
energia_multinivel = 0

## Creo una lista donde asocio los coeficientes con las sub bandas de frecuencia
subbandas = []

## Itero para cada una de las listas de coeficientes
for i in range (1, len(coefs)):

    ## Actualizo el valor de la energía agregando el total de los cuadrados de los coeficientes
    energia_multinivel += np.sum(np.square(coefs[i]))

    ## Especifico el rango de frecuencias correspondiente a la subbanda
    rango = [2 ** ( - (i + 1)) / T, 2 ** ( - i ) / T]

    ## Asocio los coeficientes con la subbanda correspondiente y su energia
    subbandas.append((rango, np.sum(np.square(coefs[len(coefs) - i])), coefs[len(coefs) - i]))

## Asigno el rango correspondiente a los coefs de aproximación
rango = [0, 2 ** ( - (len(coefs))) / T]

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
    rangos.append("{} - {}".format(banda[0][0], banda[0][1]))

    ## Agrego la energia relativa a la lista de energias
    energias.append(banda[1] / energia_multinivel)

## Hago el gráfico de barras correspondiente
plt.bar(np.array(rangos), np.array(energias))
plt.xticks(rotation = 20)
plt.xlabel("Frecuencias (Hz)")
plt.ylabel("Energía Relativa")
plt.show()