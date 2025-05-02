## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatos import *
from LecturaDatosPacientes import *
from ParametrosGaitPy import *
from SegmentacionGaitPy import *
from Escalogramas import *
from Escalado import *

## ------------------------------------ GENERACIÓN DE ESCALOGRAMAS --------------------------------------

## La idea es poder generar y guardar los escalogramas para todos los pacientes presentes en la base de datos
## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes:

    ## Creo un bloque try catch en caso de que ocurra algún error en el procesamiento
    try:

        ## Hago la lectura de los datos del registro de marcha del paciente
        data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona)

        ## Hago el cálculo de los parámetros de marcha
        features, acc_VT = ParametrosMarcha(id_persona, data, periodoMuestreo, pacientes)

        ## Hago la segmentación de la marcha
        pasos = Segmentacion(features, periodoMuestreo, acc_VT)

        ## Hago el cálculo de los escalogramas correspondientes
        escalogramas_segmentos, directorio_muestra, nombre_base_segmento = Escalogramas(id_persona, tiempo, pasos, cant_muestras, acel, gyro, periodoMuestreo)

        ## Hago el escalado y el guardado correspondiente de los escalogramas
        Escalado(escalogramas_segmentos, directorio_muestra, nombre_base_segmento)

        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))
    
    ## En caso de que ocurra un error en el procesamiento
    except:

        ## Sigo con el paciente que sigue
        continue