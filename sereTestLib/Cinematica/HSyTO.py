## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from Segmentacion import *

## ------------------------------------ DETECCIÓN TERMINAL CONTACT -------------------------------------

## Uso el método de los papers de Rafael C. Gonzalez
## Filtrado FIR pasabajos de orden 30 con frecuencia de corte 2.5Hz
lowpass = signal.firwin(numtaps = 30, cutoff = 2.5, pass_zero = 'lowpass', fs = 1 / periodoMuestreo)

## Aplico el filtro anterior a la aceleración vertical
acel_lowpass = signal.convolve(acel[:,1], lowpass, mode = 'same')

minimos = signal.argrelmin(acel_lowpass)[0]

## Creo una lista donde guardo los mínimos por detección de Toe Offs
minimos_to = []

## Itero para cada uno de los mínimos detectados
for i in range (len(minimos)):

    ## Itero para cada uno de los ceros TO detectados por el método ZC
    for j in range (len(ceros_to)):

        ## En caso de que exista un cero TO detectado con ZC que esté lo suficientemente cerca, tomo ese punto como un toe off
        if ceros_to[j] * 0.99 < minimos[i] < ceros_to[j] * 1.01:

            ## Lo agrego a la lista
            minimos_to.append(minimos[i])

            ## Finalizo el sub-bucle para pasar al siguiente TO potencial
            break

plt.plot(-acel[:,2])
plt.plot(acel_lowpass - constants.g)
plt.plot(ceros_to, np.zeros(len(ceros_to)), "o", label = 'Toe Offs')
plt.plot(minimos_to, (acel_lowpass - constants.g)[minimos_to], "x", label = 'Minimos')
plt.show()
