## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosIniciales import *
from ContactosTerminales import *
from ParametrosGaitPy import *
from SegmentacionGaitPy import Segmentacion as SegmentacionGaitPy
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid

## -------------------------------------- SEGMENTACIÓN DE MARCHA ---------------------------------------

def Segmentacion(picos_sucesivos, toe_offs, muestras_paso, periodoMuestreo, acc_AP_norm, gyro = None):

    ## ----------------------------------------- CONJUNTO DE PASOS -----------------------------------------

    ## Creo una lista donde voy a almacenar todos los pasos
    pasos = []

    ## Genero una lista en donde voy a guardar los pasos defectuosos
    ## Un paso defectuoso va a ser aquel cuyos dos contactos iniciales (ICs) detectados estén separados por una distancia alejada de la media
    ## En un paso defectuoso tengo dos posibilidades:
    ## i) En caso de que hayan más muestras que la media, puede ocurrir que hayan otros ICs en el medio que no hayan sido detectados
    ## ii) En caso que hayan menos muestras que la media, hay uno de los dos picos que no se trata de un IC 
    pasos_defectuosos = []

    ## Genero una lista en la cual voy a guardar los giros detectados
    giros = []

    ## Itero para cada uno de los picos detectados
    for i in range (len(picos_sucesivos) - 1):

        ## Genero la variable donde guardo el Toe Off a incluír por defecto en 0
        toe_off = 0

        ## Busco el Toe Off que haya detectado para asociarlo al paso
        for picoTO in toe_offs:
            
            ## En caso que el Toe Off esté entre los dos ICs
            if picos_sucesivos[i] < picoTO < picos_sucesivos[i + 1]:

                ## Me lo guardo
                toe_off = picoTO
        
        ## Entonces el par de picos me está diciendo que ahí hay un paso y entonces me lo guardo
        ## Me guardo también el Toe Off que haya detectado entre los dos pasos
        pasos.append({'IC': (picos_sucesivos[i], picos_sucesivos[i + 1]),'TC': toe_off})
        
        ## En caso de que la distancia entre los picos sea mayor a la esperada
        if not (0.7 * muestras_paso < picos_sucesivos[i + 1] - picos_sucesivos[i] < 1.3 * muestras_paso):

            ## Agrego el paso a la lista de pasos defectuosos
            pasos_defectuosos.append({'IC': (picos_sucesivos[i], picos_sucesivos[i + 1])})
        
        ## Hago la integración usando trapezoidal de la señal del giroscopio vertical segmentada en el paso
        angulos_y = cumulative_trapezoid(gyro[:, 1][picos_sucesivos[i] : picos_sucesivos[i + 1]], dx = periodoMuestreo, initial = 0)

        ## En caso de que la diferencia entre el ángulo máximo y el mínimo sea mayor a 45 quiere decir que estoy en presencia de un giro
        ## Hay que tener cuidado porque el umbral está en rad/s
        if np.max(np.abs(angulos_y)) - np.min(np.abs(angulos_y)) > 35:

            ## En caso de que el intervalo detectado anterior se trate del mismo giro
            if len(giros) > 0 and giros[-1][1] == picos_sucesivos[i]:

                ## Creo un intervalo que abarque todo el giro
                giros[-1][1] = picos_sucesivos[i + 1]
            
            ## En caso de que tenga un sólo tramo de giro
            else:

                ## Me guardo entonces los tiempos asociados al giro
                giros.append([picos_sucesivos[i], picos_sucesivos[i + 1]])

    ## ---------------------------------- TRATAMIENTO DE PASOS DEFECTUOSOS ---------------------------------

    ## Itero para cada uno de los pasos defectuosos
    ## Éste bloque lo puedo mover dentro del for de arriba para reducir iteraciones, lo dejo así para que quede más prolijo
    for i in range (len(pasos_defectuosos)):

        ## Calculo la longitud del paso defectuoso en cantidad de muestras
        long_paso_defectuoso = pasos_defectuosos[i]['IC'][1] - pasos_defectuosos[i]['IC'][0]

        ## En caso de que la longitud del paso defectuoso sea mayor a la cantidad promedio de muestras
        if long_paso_defectuoso > muestras_paso:

            ## Mi tarea acá es encontrar posibles potenciales pasos intermedios que no fueron detectados
            picos_interpolados = find_peaks(acc_AP_norm[pasos_defectuosos[i]['IC'][0] : pasos_defectuosos[i]['IC'][1]])

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
    
    ## Retorno una lista de diccionarios con los pasos detectados
    return pasos, duraciones_pasos, giros