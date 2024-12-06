## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosInicialesM2 import *
from ContactosTerminalesM2 import *

## --------------------------------------- GRAFICACIÓN DE DATOS ----------------------------------------

# ## Graficación de Heel Strikes y Toe Offs tomando la aceleración vertical filtrada
# plt.plot(ceros_hs, np.zeros(len(ceros_hs)), "x", label = 'Heel Strikes')
# plt.plot(ceros_to, np.zeros(len(ceros_to)), "o", label = 'Toe Offs')
# plt.plot(filtrada, label = "Aceleracion Vertical Filtrada")
# plt.legend()
# plt.show()

# ## Graficación de Heel Strikes y Toe Offs tomando la aceleración anteroposterior
# plt.plot(ceros_hs, np.zeros(len(ceros_hs)), "x", label = 'Heel Strikes')
# plt.plot(ceros_to, np.zeros(len(ceros_to)), "o", label = 'Toe Offs')
# plt.plot(-acel[:,2], label = "Aceleracion Anteroposterior")
# plt.legend()
# plt.show()

# print("Tiempo Paso Promedio: {}".format(np.mean(np.diff(ceros_hs)) * periodoMuestreo))
# print("Tiempo Paso Desviación Estándar: {}".format(np.std(np.diff(ceros_hs)) * periodoMuestreo))
# print("Tiempo Paso Mediana: {}".format(np.median(np.diff(ceros_hs)) * periodoMuestreo))

## ------------------------------------ DETECCIÓN PRIMER EVENTO ----------------------------------------

## En caso que el primer cero del HS se encuentre antes del primer cero del TO
if ceros_hs[0] < ceros_to[0]:

    ## Entonces el primer evento va a ser un Heel Strike
    primer_evento = 'hs'

## En caso que el primer cero del TO se encuentre antes del primer cero del HS
else:

    ## Entonces el primer evento va a ser un Toe Off
    primer_evento = 'to'

## ------------------------------- PROPORCIÓN ESTANCIA DOBLE Y SIMPLE ----------------------------------

## Creo una lista donde voy a guardar las proporciones de estancia doble y simple
proporciones_paso = []

## Itero para cada uno de los eventos que tengo detectados
for i in range (len(ceros_hs) - 1):

    ## En caso que el primer evento sea un HS
    if primer_evento == 'hs':

        ## Calculo la proporción del paso como DoubleStance/SingleStance = (TO[i]-HS[i])/(HS[i+1]-TO[i])
        proporcion = (ceros_to[i] - ceros_hs[i]) / (ceros_hs[i + 1] - ceros_to[i])

        ## Agrego la proporción calculada a la lista de proporciones de paso
        proporciones_paso.append(proporcion)
    
    ## En caso que el primer evento sea un TO
    else:

        ## Calculo la proporción del paso como DoubleStance/SingleStance = (TO[i+1]-HS[i])/(HS[i]-TO[i])
        proporcion = (ceros_to[i + 1] - ceros_hs[i]) / (ceros_hs[i] - ceros_to[i])

        ## Agrego la proporción calculada a la lista de proporciones de paso
        proporciones_paso.append(proporcion)

## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

## Creo una lista donde voy a guardar los pasos detectados por Zero Crossing
pasos = []

## Itero para cada uno de los eventos HS que tengo detectados
for i in range (len(ceros_hs) - 1):

    ## En caso de que el primer evento haya sido un Heel Strike
    if primer_evento == 'hs':

        ## Defino el paso como un diccionario donde tengo [HS[i], HS[i+1]] como los ICs y [TO[i]] como el TC
        ## Tengo que agregar los valores de los HS y TO para luego poder hacer la segmentación de la señal de posición al medir el paso
        paso_zc = {'IC': (heel_strikes[i], heel_strikes[i + 1]),'TC': toe_offs[i]}
    
    ## En caso que el primer evento haya sido un Toe Off
    else:

        ## Defino el paso como un diccionario donde tengo [HS[i], HS[i+1]] como los ICs y [TO[i+1]] como el TC
        ## Tengo que agregar los valores de los HS y TO para luego poder hacer la segmentación de la señal de posición al medir el paso
        paso_zc = {'IC': (heel_strikes[i], heel_strikes[i + 1]),'TC': toe_offs[i + 1]}
    
    ## Agrego el paso detectado a la lista de pasos
    pasos.append(paso_zc)

## ----------------------------------------- DURACIÓN DE PASOS -----------------------------------------

## Creo una lista donde voy a almacenar las muestras entre todos los pasos
muestras_pasos = []

## Creo una lista en donde voy a almacenar las duraciones de todos los pasos
duraciones_pasos = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):
    
    ## Calculo la diferencia entre ambos valores de la tupla en términos temporales
    diff_pasos = pasos[i]['IC'][1] - pasos[i]['IC'][0]

    ## Almaceno la diferencia de muestras en la lista de muestras entre pasos
    muestras_pasos.append(diff_pasos)

    ## Almaceno la diferencia temporal entre los pasos en otra lista
    duraciones_pasos.append(diff_pasos * periodoMuestreo)

## --------------------------------------- TIEMPO ENTRE IC Y TC ----------------------------------------

## Creo una lista donde almaceno las distancias entre ICs y TCs expresado en muestras
dist_IC_TC = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):
    
    ## Calculo la distancia entre IC y TC
    dist = pasos[i]['TC'] - pasos[i]['IC'][0]

    ## Agrego la distancia a la lista
    dist_IC_TC.append(dist)

## Genero la lista de tiempos
dist_IC_TC_tiempo = np.multiply(periodoMuestreo, dist_IC_TC)

## --------------------------------------- CALCULO DOBLE ESTANCIA --------------------------------------

## Genero una lista vacía donde voy a calcular las proporciones de doble estancia en un paso
doble_estancia = []

## Itero para cada uno de los pasos detectados
for i in range (len(pasos)):

    ## Calculo la proporcion de la doble estancia
    doble_estancia_paso = (pasos[i]['TC'] - pasos[i]['IC'][0]) / (pasos[i]['IC'][1] - pasos[i]['IC'][0])

    ## En caso que la doble estancia tenga un valor no permitido, que se saltee ésta parte
    if abs(doble_estancia_paso) > 1:

        ## Se saltea ésta iteración
        continue

    ## La agrego a la lista
    doble_estancia.append(doble_estancia_paso)