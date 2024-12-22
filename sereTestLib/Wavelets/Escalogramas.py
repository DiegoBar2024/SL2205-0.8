## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

from SegmentacionM1 import *

## --------------------------------------------- PARÁMETROS --------------------------------------------

## Escalas de las wavelets a utilizar
## Se recuerda que la frecuencia me queda f = f_muestreo / escala
## Ésto me implica que a escalas más pequeñas tengo frecuencias más grandes
escalas = np.arange(50, 150, 1)

## Tipo de wavelet a utilizar. En principio Morlet Wavelet Compleja
## Parámetro B (Ancho de banda): 1.5 Hz
## Parámetro C (Frecuencia Central): 1 Hz
wavelet = 'cmor1.5-1'

## ------------------------------------------- SEGMENTACIÓN --------------------------------------------

## Creo una matriz donde se van a guardar los segmentos de las señales de acelerómetros y giroscopios según los pasos detectados
## La indexación de <<matriz_segmentada>> en [Dimension1, Dimension2, Dimension3] se indica del siguiente modo:
##  i)   Dimension1: Indica el número de paso
##  ii)  Dimension2: Indica el canal
##  iii) Dimension3: Indica el instante temporal
matriz_segmentada = np.zeros(len(pasos), dtype = object)

## Creo un vector en donde voy guardando las extensiones en muestras correspondientes a cada paso
extensiones_pasos = []

## Ahora itero para cada uno de los pasos que tengo segmentados
for i in range (len(pasos)):

    ## Le doy a la extensión un valor predefinido de 400 muestras (aproximadamente 2 segundos)
    extension = 400

    ## En caso de que la extensión sea mayor al tiempo que tengo, lo seteo en el limite
    if extension > pasos[i]['IC'][0]:

        ## Actualizo el valor de la extensión a realizar
        extension = pasos[i]['IC'][0]

    ## Filas de <<matriz>>: Son cada uno de los 6 canales que tengo. Es decir señales de acelerómetros y giroscopios
    ## Columnas de <<matriz>>: Son cada uno de los instantes temporales que componen el paso detectado entre dos ICs
    ## Para mitigar los efectos de borde se le agrega una extensión en ambos lados del intervalo [IC[i] - e, IC[i+1] + e]
    matriz = np.array([ acel[:,0][(pasos[i]['IC'][0] - extension) : (pasos[i]['IC'][1] + extension)],
                        acel[:,1][(pasos[i]['IC'][0] - extension) : (pasos[i]['IC'][1] + extension)],
                        acel[:,2][(pasos[i]['IC'][0] - extension) : (pasos[i]['IC'][1] + extension)],
                        gyro[:,0][(pasos[i]['IC'][0] - extension) : (pasos[i]['IC'][1] + extension)],
                        gyro[:,1][(pasos[i]['IC'][0] - extension) : (pasos[i]['IC'][1] + extension)],
                        gyro[:,2][(pasos[i]['IC'][0] - extension) : (pasos[i]['IC'][1] + extension)] ])

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

## Itero para cada uno de los pasos que tengo segmentados
for i in range (len(pasos)):

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
    matriz_escalogramas[i] = np.array([     coef1[:,extensiones_pasos[i]: coef1.shape[1] - extensiones_pasos[i]], 
                                            coef2[:,extensiones_pasos[i]: coef2.shape[1] - extensiones_pasos[i]],
                                            coef3[:,extensiones_pasos[i]: coef3.shape[1] - extensiones_pasos[i]], 
                                            coef4[:,extensiones_pasos[i]: coef4.shape[1] - extensiones_pasos[i]], 
                                            coef5[:,extensiones_pasos[i]: coef5.shape[1] - extensiones_pasos[i]], 
                                            coef6[:,extensiones_pasos[i]: coef6.shape[1] - extensiones_pasos[i]]     ])

    ## Graficación del escalograma
    data = np.abs(coef3[:,extensiones_pasos[i]: coef3.shape[1] - extensiones_pasos[i]]) ** 2
    cmap = plt.get_cmap('jet', 256)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    t = np.arange(coef3[:,extensiones_pasos[i]: coef3.shape[1] - extensiones_pasos[i]].shape[1]) * periodoMuestreo
    ax.pcolormesh(t, scales_freq, data, cmap=cmap, vmin=data.min(), vmax=data.max(), shading='auto')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title("$|CWT(t,f)|^2$")
    plt.show()