## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from Segmentacion import *

## --------------------------------------------- PARÁMETROS --------------------------------------------

## Escalas de las wavelets a utilizar
escalas = [8, 136]

## Tipo de wavelet a utilizar
wavelet = 'cmor1.5-1'

## ------------------------------------- SEGMENTACIÓN (PEAK FINDING) -----------------------------------

## Creo una matriz donde se van a guardar los segmentos de las señales de acelerómetros y giroscopios según los pasos detectados
## La indexación de <<matriz_segmentada_pf>> en [Dimension1, Dimension2, Dimension3] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica el instante temporal
matriz_segmentada_pf = np.zeros(len(pasos), dtype = object)

## Ahora itero para cada uno de los pasos que tengo segmentados
for i in range (len(pasos)):

    ## Filas de <<matriz>>: Son cada uno de los 6 canales que tengo. Es decir señales de acelerómetros y giroscopios
    ## Columnas de <<matriz>>: Son cada uno de los instantes temporales que componen el paso detectado entre dos ICs
    matriz = np.array([acel[:,0][pasos[i]['IC'][0] : pasos[i]['IC'][1]],
                            acel[:,1][pasos[i]['IC'][0] : pasos[i]['IC'][1]],
                            acel[:,2][pasos[i]['IC'][0] : pasos[i]['IC'][1]],
                            gyro[:,0][pasos[i]['IC'][0] : pasos[i]['IC'][1]],
                            gyro[:,1][pasos[i]['IC'][0] : pasos[i]['IC'][1]],
                            gyro[:,2][pasos[i]['IC'][0] : pasos[i]['IC'][1]]])

    ## Agrego la matriz al tensor de datos segmentados
    matriz_segmentada_pf[i] = matriz

## ------------------------------------- SEGMENTACIÓN (ZERO CROSSING) ----------------------------------

## Creo una matriz donde se van a guardar los segmentos de las señales de acelerómetros y giroscopios según los pasos detectados
## La indexación de <<matriz_segmentada_zc>> en [Dimension1, Dimension2, Dimension3] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica el instante temporal
matriz_segmentada_zc = np.zeros(len(pasos_zc), dtype = object)

## Ahora itero para cada uno de los pasos que tengo segmentados
for i in range (len(pasos_zc)):

    ## Filas de <<matriz>>: Son cada uno de los 6 canales que tengo. Es decir señales de acelerómetros y giroscopios
    ## Columnas de <<matriz>>: Son cada uno de los instantes temporales que componen el paso detectado entre dos ICs
    matriz = np.array([acel[:,0][pasos_zc[i]['IC'][0] : pasos_zc[i]['IC'][1]],
                            acel[:,1][pasos_zc[i]['IC'][0] : pasos_zc[i]['IC'][1]],
                            acel[:,2][pasos_zc[i]['IC'][0] : pasos_zc[i]['IC'][1]],
                            gyro[:,0][pasos_zc[i]['IC'][0] : pasos_zc[i]['IC'][1]],
                            gyro[:,1][pasos_zc[i]['IC'][0] : pasos_zc[i]['IC'][1]],
                            gyro[:,2][pasos_zc[i]['IC'][0] : pasos_zc[i]['IC'][1]]])

    ## Agrego la matriz al tensor de datos segmentados
    matriz_segmentada_zc[i] = matriz

## -------------------------------------- ESCALOGRAMAS (PEAK FINDING) ----------------------------------

## Creo una matriz en la cual para cada paso asocio un tensor tridimensional con las transformadas de wavelet
## La indexación de <<matriz_escalogramas_pf>> en [Dimension1, Dimension2, Dimension3, Dimension4] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica la escala
##  iv)  Dimension4: Indica el instante temporal
matriz_escalogramas_pf = np.zeros(len(pasos), dtype = object)

## Itero para cada uno de los pasos que tengo segmentados
for i in range (len(pasos)):

    ## Transformada de Wavelet de la aceleración en el eje x
    coef1, scales_freq = pywt.cwt(data = matriz_segmentada_pf[i][0], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la aceleración en el eje y
    coef2, scales_freq = pywt.cwt(data = matriz_segmentada_pf[i][1], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la aceleración en el eje z
    coef3, scales_freq = pywt.cwt(data = matriz_segmentada_pf[i][2], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la señal de giroscopio respecto del eje x
    coef4, scales_freq = pywt.cwt(data = matriz_segmentada_pf[i][3], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)
    
    ## Transformada de Wavelet de la señal de giroscopio respecto del eje y
    coef5, scales_freq = pywt.cwt(data = matriz_segmentada_pf[i][4], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)
    
    ## Transformada de Wavelet de la señal de giroscopio respecto del eje z
    coef6, scales_freq = pywt.cwt(data = matriz_segmentada_pf[i][5], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Agrego al tensor de escalogramas una pila con los 6 coeficientes
    matriz_escalogramas_pf[i] = np.array([coef1, coef2, coef3, coef4, coef5, coef6])

    # fig, axs = plt.subplots(1, 1)
    # pcm = axs.pcolormesh(np.abs(coef2[:-1, :-1]))
    # axs.set_yscale("log")
    # axs.set_xlabel("Time (s)")
    # axs.set_ylabel("Frequency (Hz)")
    # axs.set_title("Continuous Wavelet Transform (Scaleogram)")
    # fig.colorbar(pcm, ax=axs)
    # plt.show()

## -------------------------------------- ESCALOGRAMAS (ZERO CROSSING) ---------------------------------

## Creo una matriz en la cual para cada paso asocio un tensor tridimensional con las transformadas de wavelet
## La indexación de <<matriz_escalogramas_zc>> en [Dimension1, Dimension2, Dimension3, Dimension4] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica la escala
##  iv)  Dimension4: Indica el instante temporal
matriz_escalogramas_zc = np.zeros(len(pasos), dtype = object)

## Itero para cada uno de los pasos que tengo segmentados
for i in range (len(pasos)):

    ## Transformada de Wavelet de la aceleración en el eje x
    coef1, scales_freq = pywt.cwt(data = matriz_segmentada_zc[i][0], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la aceleración en el eje y
    coef2, scales_freq = pywt.cwt(data = matriz_segmentada_zc[i][1], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la aceleración en el eje z
    coef3, scales_freq = pywt.cwt(data = matriz_segmentada_zc[i][2], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Transformada de Wavelet de la señal de giroscopio respecto del eje x
    coef4, scales_freq = pywt.cwt(data = matriz_segmentada_zc[i][3], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)
    
    ## Transformada de Wavelet de la señal de giroscopio respecto del eje y
    coef5, scales_freq = pywt.cwt(data = matriz_segmentada_zc[i][4], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)
    
    ## Transformada de Wavelet de la señal de giroscopio respecto del eje z
    coef6, scales_freq = pywt.cwt(data = matriz_segmentada_zc[i][5], scales = escalas, wavelet = wavelet, sampling_period = periodoMuestreo)

    ## Agrego al tensor de escalogramas una pila con los 6 coeficientes
    matriz_escalogramas_zc[i] = np.array([coef1, coef2, coef3, coef4, coef5, coef6])