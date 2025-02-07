## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosInicialesM1 import *
from ContactosTerminalesM1 import *

## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

## Creo una lista donde voy a almacenar todos los pasos
pasos = []

## Itero para cada uno de los picos detectados
for i in range (len(picos_sucesivos) - 1):

    ## En caso que la distancia entre picos esté en un rango aceptable, concluyo que ahí se habrá detectado un paso
    if (0.7 * muestras_paso < picos_sucesivos[i + 1] - picos_sucesivos[i] < 1.3 * muestras_paso):

        ## Genero la variable donde guardo el Toe Off a incluír por defecto en 0
        toe_off = 0

        ## Busco el Toe Off que haya detectado para asociarlo al paso
        for picoTO in toe_offs_min:
            
            ## En caso que el Toe Off esté entre los dos ICs
            if picos_sucesivos[i] < picoTO < picos_sucesivos[i + 1]:

                ## Me lo guardo
                toe_off = picoTO
        
        ## Entonces el par de picos me está diciendo que ahí hay un paso y entonces me lo guardo
        ## Me guardo también el Toe Off que haya detectado entre los dos pasos
        pasos.append({'IC': (picos_sucesivos[i], picos_sucesivos[i + 1]),'TC': toe_off})
    
    # ## En caso de que la distancia entre los picos sea mayor a la esperada
    # else:

    #     ## Grafica de la señal recortada en ese tramo
    #     plt.plot(acc_AP_norm[picos_sucesivos[i] : picos_sucesivos[i + 1]])
    #     plt.title("Cantidad de muestras: {}\nCantidad de muestras por paso: {}".format(picos_sucesivos[i + 1] - picos_sucesivos[i], muestras_paso))
    #     plt.show()

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

## ---------------------------------------- CANTIDAD DE PASOS ------------------------------------------

## Me guardo en una variable la cantidad total de pasos
cantidad_pasos = len(pasos)

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