## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Normalizacion import Normalizacion
from DeteccionPicos import *
from Fourier import TransformadaFourier
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from harm_analysis import *
from control import *
from scipy import constants

from LecturaDatos import *

## ------------------------------------- DETECCIÓN CONTACTOS TERMINALES ------------------------------------

def ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False):

    ## ------------------------------------- ANÁLISIS EN FRECUENCIA ----------------------------------------

    ## Señal de aceleración vertical (resto la gravedad que me da una buena aproximación)
    acc_VT = acel[:,1] - constants.g

    ## Filtro de Butterworth pasabanda de rango [0.5, 2.5] Hz para que me detecte el armónico fundamental
    ## A partir del armónico fundamental obtengo la cadencia promedio de la marcha
    filtro = signal.butter(N = 4, Wn = [0.5, 2.5], btype = 'bandpass', fs = 1 / periodoMuestreo, output = 'sos')

    ## Aplico el filtro anterior a la aceleración anteroposterior
    acel_filtrada = signal.sosfiltfilt(filtro, acc_VT)

    ## Hago la normalizacion de la aceleración vertical filtrada
    acc_VT_norm = Normalizacion(acel_filtrada)

    ## Se obtiene el espectro de toda la señal completa aplicando la transformada de Fourier
    ## Ya que es una señal real se cumple la simetría conjugada
    (frecuencias, transformada) = TransformadaFourier(acel_filtrada, periodoMuestreo, plot = False)

    ## Cálculo de los coeficientes de Fourier en semieje positivo
    coefs = (2 / acel_filtrada.shape[0]) * np.abs(transformada)[:acel_filtrada.shape[0]//2]

    ## Calculo de las frecuencias en el semieje positivo
    frecs = frecuencias[:acel_filtrada.shape[0]//2]

    ## Elimino la componente de contínua de la señal
    coefs[0] = 0

    ## Determino la posición en la que se da el máximo. 
    ## ESTOY ASUMIENDO QUE EL MÁXIMO SE DA EN LA COMPONENTE FUNDAMNENTAL (no tiene porque ocurrir!)
    ## Ésta será considerada como la frecuencia fundamental de la señal
    pos_maximo = np.argmax(coefs)

    ## Frecuencia fundamental de la señal
    ## La frecuencia fundamental de la señal en las aceleraciones CC (craneo-cervical) y AP (antero-posterior) son iguales.
    ## Se podría interpretar como la frecuencia fundamental de los pasos en la marcha de la persona
    frec_fund = frecs[pos_maximo]

    ## Estimación del tiempo de paso en base al análisis en frecuencia
    ## La estimación se hace en base al armónico de mayor amplitud de la señal
    tiempo_paso_frecuencia = 1 / frec_fund

    ## Calculo la cantidad promedio de muestras que tengo en mi señal por cada paso
    muestras_paso = tiempo_paso_frecuencia / periodoMuestreo

    ## -------------------------------------- DETECCIÓN PRIMER PICO ----------------------------------------

    ## Especifico un umbral predefinido para la detección de picos
    umbral = 0.1

    ## Hago la detección de picos para un umbral predefinido
    picos = DeteccionPicos(acc_VT_norm, umbral = umbral)

    ## Me tomo un entorno de 0.3 * P hacia delante centrado en el primer pico detectado
    ## Ésto lo hago porque puede pasar en algún caso que el primer pico detectado no sea el correcto
    rango = picos[0] + np.array([0, 0.3 * muestras_paso])

    ## Rango de posibles primeros picos
    picos_rango = picos[picos < 0.3 * muestras_paso + picos[0]]

    ## Obtengo el índice del valor correspondiente al pico máximo
    ind_pico_maximo = np.argmax(acc_VT_norm[picos_rango])

    ## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
    pico_maximo_inicial = picos_rango[ind_pico_maximo]

    ## ------------------------------------ DETECCIÓN PICOS SUCESIVOS --------------------------------------

    ## Creo una lista donde voy almacenando las posiciones de los picos sucesivos
    ## En ésta lista se va a almacenar el primer pico detectado por el algoritmo previo
    picos_sucesivos = [pico_maximo_inicial]

    ## Creo una variable donde voy guardando el valor del indice de los picos sucesivos. Lo inicializo en 0
    ind_picos_sucesivos = 0

    ## Mientras que el rango no supere la longitud de la señal, que siga iterando
    while (rango[0] < cant_muestras):

        ## Se calcula el rango de separación donde se espera que esté el próximo pico.
        ## Empíricamente se escoge [0.7 * P, 1.3 * P] donde P sería la cantidad de muestras por pico (paper de Zhao)
        rango = picos_sucesivos[ind_picos_sucesivos] + np.array([0.7 * muestras_paso, 1.3 * muestras_paso])

        ## Aumento en una unidad el valor del índice
        ind_picos_sucesivos += 1

        ## Mientras que no haya picos detectados en el rango, sigo iterando éste sub bucle
        while True:

            ## Hago la segmentación de la señal de aceleración AP en éste rango. Hago la conversión de los umbrales a enteros
            segment_rango = acc_VT_norm[int(rango[0]) : int(rango[1])]

            ## Hago la detección de picos en éste rango de señal de aceleración AP con el umbral preconfigurado
            ## Hago la detección de picos para un umbral predefinido
            picos_rango = DeteccionPicos(segment_rango, umbral = umbral)

            ## Debo ponerle dos condiciones para que el bucle se pare:
            # i) En caso de que haya picos detectados, rompo el bucle para interpolar el pico que fue detectado
            # ii) En caso de que el extremo izquierdo del rango sea mayor a la longitud de la señal
            if len(picos_rango) > 0  or rango[0] > cant_muestras:

                    ## Ruptura de bucle
                    break

            ## Seteo la referencia al pico previo sumado 0.7 * P en caso de que no haya ningún pico detectado en el rango actual
            rango = 0.7 * muestras_paso + rango

        ## En caso que el bucle haya salido porque se detectaron picos
        if len(picos_rango) > 0:

            ## Obtengo el valor se la señal de aceleración AP para éstos picos
            segment_picos = segment_rango[picos_rango]

            ## Obtengo el índice del valor correspondiente al pico máximo
            ind_pico_maximo = np.argmax(segment_picos)

            ## Como están en el mismo orden puedo indexar el pico máximo en la lista de los picos detectados en el rango
            pico_maximo = picos_rango[ind_pico_maximo]

            ## Agrego a la lista de picos sucesivos el pico detectado
            ## Recuerdo que debo sumarle al pico detectado el primer elemento del rango para llevarlo a la escala real
            picos_sucesivos.append(pico_maximo + int(rango[0]))

    ## --------------------------- PROCESADO DE LA SEÑAL DE ACELERACIÓN VERTICAL ---------------------------

    ## Uso el método de los papers de Rafael C. Gonzalez
    ## Filtrado FIR pasabajos de orden 30 con frecuencia de corte 12.5Hz (frecuencia de corte tuneada manualmente)
    ## Pruebo subir un poco la frecuencia de corte del FIR para poder detectar de mejor manera los mínimos locales donde estarían los TO
    lowpass = signal.firwin(numtaps = 30, cutoff = 12.5, pass_zero = 'lowpass', fs = 1 / periodoMuestreo)

    ## Aplico el filtro anterior a la aceleración vertical
    acel_lowpass = signal.convolve(acel[:,1], lowpass, mode = 'same')

    ## --------------------------------- DETECCIÓN DE CONTACTOS TERMINALES ---------------------------------

    ## Creo una lista vacía en donde voy a guardar los eventos de Toe Off detectados
    toe_offs = []

    ## Itero para cada uno de los pasos que tengo detectados
    for i in range (len(picos_sucesivos) - 1):

        ## Hago la segmentación de la señal entre dos picos máximos
        segmento = acel_lowpass[picos_sucesivos[i] : picos_sucesivos[i + 1]]

        ## Hago la detección de todos los mínimos locales en el segmento
        minimos_segmento = signal.argrelmin(segmento)[0]

        ## Me quedo con el primer mínimo el cual yo supongo que va a estar asociado al evento de TOE OFF
        toe_off = minimos_segmento[0]

        ## Agrego el TO detectado a la lista de toe offs junto con el desfasaje de muestra según el pico de la vertical
        toe_offs.append(toe_off + picos_sucesivos[i])

    ## ------------------------------- GRAFICACIÓN DE CONTACTOS TERMINALES ---------------------------------

    ## En caso de que quiera graficar los datos
    if graficar:

        # ## Hago la graficación del espaciamiento entre Toe Offs
        # plt.scatter(x = np.arange(start = 0, stop = len(np.diff(toe_offs))), y = np.diff(toe_offs) * periodoMuestreo)
        # plt.show()

        ## Hago la graficación de los Toe Offs junto con la señal de aceleración vertical filtrada con el FIR
        plt.plot(acel_lowpass, label = 'Aceleración Vertical')
        plt.plot(picos_sucesivos, acel_lowpass[picos_sucesivos], "x", label = 'Contactos Iniciales')
        plt.plot(toe_offs, acel_lowpass[toe_offs], "o", label = 'Contactos Terminales')
        plt.legend()
        plt.show()
    
    ## Retorno la lista con los eventos de Toe Off detectados
    return toe_offs