## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

from ContactosIniciales import *
from ContactosTerminales import *
from ParametrosGaitPy import *
from LongitudPasoM1 import LongitudPasoM1
from Segmentacion import Segmentacion
import sys
import pandas as pd
import json
from LecturaDatosPacientes import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')

## ------------------------------------- CÁLCULO DE ESTADÍSTICAS ---------------------------------------

## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Itero para cada uno de los identificadores de los pacientes
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Hago la lectura de los datos del registro de marcha del paciente
        data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona)

        ## Cálculo de contactos iniciales
        contactos_iniciales, muestras_paso, acc_AP_norm, frec_fund = ContactosIniciales(acel, cant_muestras, periodoMuestreo, graficar = False)

        ## Cálculo de contactos terminales
        contactos_terminales = ContactosTerminales(acel, cant_muestras, periodoMuestreo, graficar = False)

        ## Hago la segmentación de la marcha
        pasos, duraciones_pasos = Segmentacion(contactos_iniciales, contactos_terminales, muestras_paso, periodoMuestreo, acc_AP_norm, gyro)

        ## Cálculo de parámetros de marcha
        pasos_numerados, frecuencias, velocidades, long_pasos_m1, coeficientes_m1 = LongitudPasoM1(pasos, acel, tiempo, periodoMuestreo, frec_fund, duraciones_pasos)

        ## Hago la lectura del archivo JSON previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Estadisticas.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            estadisticas_marcha = json.load(openfile)

        ## Agrego un nuevo elemento al diccionario de estadísticas
        ## La clave va a ser el ID de la persona cuyo registro se está procesando
        ## El valor va a ser una tupla con los valores medios de (Longitud Promedio de Paso (m), Duración Promedio de Paso (s), Cadencia Promedio (pasos / s), Velocidad Promedio (m/s))
        estadisticas_marcha[str(id_persona)] = (np.mean(long_pasos_m1), np.mean(duraciones_pasos), np.mean(frecuencias), np.mean(velocidades))

        ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica/Estadisticas.json", "w") as outfile:

            ## Escribo el diccionario actualizado
            json.dump(estadisticas_marcha, outfile)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))

    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue

