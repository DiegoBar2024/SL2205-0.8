## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from LecturaDatos import *
from ContactosIniciales import *
from ContactosTerminales import *
from Segmentacion import *
from LongitudPasoM1 import *
from LongitudPasoM2 import *
import json

## ------------------------------------- OPTIMIZACIÓN DEL MODELO ---------------------------------------

def Optimizacion(long_pasos_m1, coeficientes_m1, long_pasos_m2, coeficientes_m2, long_pie, longitud_pasos, id_persona):

    ## ------------------------------------ CÁLCULO DE ERROR (MÉTODO I) ------------------------------------

    ## Construyo un vector del tamaño de la cantidad de pasos detectados cuyos valores sean igual a la longitud especificada anteriormente
    pasos_control_m1 = longitud_pasos * np.ones(np.size(long_pasos_m1))

    ## Calculo la señal de error de la longitud de los pasos calculada con la longitud de control
    error_m1 = abs(pasos_control_m1 - long_pasos_m1)

    ## ------------------------------------- OPTIMIZACIÓN (MÉTODO I) ---------------------------------------

    ## Calculo el error cuadrático medio actual de los pasos tomados
    error_medio_m1 = np.sum(np.square(error_m1)) / len(error_m1)

    ## Calculo el valor óptimo del factor de corrección usando minimización de error de minimos cuadrados (ver análisis teórico)
    optimo_m1 = (np.dot(pasos_control_m1, coeficientes_m1)) / (np.sum(np.square(coeficientes_m1)))

    ## ----------------------------------- CÁLCULO DE ERROR (MÉTODO II) ------------------------------------

    ## Construyo un vector del tamaño de la cantidad de pasos detectados cuyos valores sean igual a la longitud especificada anteriormente
    pasos_control_m2 = longitud_pasos * np.ones(np.size(long_pasos_m2))

    ## Calculo la señal de error de la longitud de los pasos calculada con la longitud de control
    error_m2 = abs(pasos_control_m2 - long_pasos_m2)

    ## ------------------------------------- OPTIMIZACIÓN (MÉTODO II) --------------------------------------

    ## Igual a la optimización con el método I aplico un modelo de mínimización de error cuadrático medio
    ## Calculo el error cuadrático medio actual de los pasos tomados
    error_medio_m2 = np.sum(np.square(error_m2)) / len(error_m2)

    ## Calculo el valor óptimo del factor de corrección usando minimización de error de minimos cuadrados (ver análisis teórico)
    optimo_m2 =  (np.sum(np.subtract(pasos_control_m2, coeficientes_m2))) / (long_pie * len(pasos_control_m2))

    ## ----------------------------------- GUARDADO DE DATOS DEL MÉTODO I ----------------------------------------

    ## Hago la lectura del archivo JSON previamente existente
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/OptimizacionM1.json", 'r') as openfile:

        # Cargo el valor existente del JSON
        dicc_optim_M1 = json.load(openfile)

    ## En caso de que la persona ya se encuentre ingresada en la base de datos
    if str(id_persona) in list(dicc_optim_M1.keys()):

        ## Se agrega el valor óptimo del parámetro para el método I a la lista correspondiente
        dicc_optim_M1[str(id_persona)].append(optimo_m1)
    
    ## En caso de que la persona no se encuentre ingresada en la base de datos
    else:

        dicc_optim_M1[str(id_persona)] = [optimo_m1]
    
    ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/OptimizacionM1.json", "w") as outfile:

        ## Escribo el diccionario actualizado
        json.dump(dicc_optim_M1, outfile)
    
    ## ---------------------------------- GUARDADO DE DATOS DEL MÉTODO Ii ----------------------------------------

    ## Hago la lectura del archivo JSON previamente existente
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/OptimizacionM2.json", 'r') as openfile:

        # Cargo el valor existente del JSON
        dicc_optim_M2 = json.load(openfile)

    ## En caso de que la persona ya se encuentre ingresada en la base de datos
    if str(id_persona) in list(dicc_optim_M2.keys()):

        ## Se agrega el valor óptimo del parámetro para el método I a la lista correspondiente
        dicc_optim_M2[str(id_persona)].append(optimo_m2)
    
    ## En caso de que la persona no se encuentre ingresada en la base de datos
    else:

        dicc_optim_M2[str(id_persona)] = [optimo_m2]
    
    ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir
    with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/OptimizacionM2.json", "w") as outfile:

        ## Escribo el diccionario actualizado
        json.dump(dicc_optim_M2, outfile)