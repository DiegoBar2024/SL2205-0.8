## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from sereTestLib.Cinematica.SegmentacionM3 import *

## --------------------------- PROCESADO DE LA SEÑAL DE ACELERACIÓN VERTICAL ---------------------------

## Uso el método de los papers de Rafael C. Gonzalez
## Filtrado FIR pasabajos de orden 30 con frecuencia de corte 2.5Hz
## Pruebo subir un poco la frecuencia de corte del FIR para poder detectar de mejor manera los mínimos locales donde estarían los TO
lowpass = signal.firwin(numtaps = 30, cutoff = 12.5, pass_zero = 'lowpass', fs = 1 / periodoMuestreo)

## Aplico el filtro anterior a la aceleración vertical
acel_lowpass = signal.convolve(acel[:,1], lowpass, mode = 'same')

## --------------------------------- DETECCIÓN DE CONTACTOS TERMINALES ---------------------------------

## Creo una lista vacía en donde voy a guardar los eventos de Toe Off detectados
toe_offs_min = []

## Itero para cada uno de los pasos que tengo detectados
for i in range (len(pasos)):

    ## Hago la segmentación de la señal entre dos picos máximos
    segmento = acel_lowpass[pasos[i]['IC'][0] : pasos[i]['IC'][1]]

    ## Hago la detección de mínimos locales en el segmento
    minimos_segmento = signal.argrelmin(segmento)[0]

    ## Me quedo con el primer mínimo el cual yo sé que va a estar asociado al evento de TOE OFF
    toe_off = minimos_segmento[0]

    ## Agrego el TO detectado a la lista de toe offs
    toe_offs_min.append(toe_off + pasos[i]['IC'][0])

    # plt.plot(np.diff(segmento))
    # plt.plot(toe_off, np.diff(segmento)[toe_off], 'x', label = 'Toe Off')
    # plt.legend()
    # plt.show()

GraficacionPicos(acel_lowpass - constants.g, toe_offs_min)

## ------------------------------- GRAFICACIÓN DE CONTACTOS TERMINALES ---------------------------------

## Hago la graficación del espaciamiento entre Toe Offs
plt.scatter(x = np.arange(start = 0, stop = len(np.diff(toe_offs_min))), y = np.diff(toe_offs_min) * periodoMuestreo)
plt.show()