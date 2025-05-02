## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
from scipy.signal import *
import numpy as np
import json
from FuncionesEnergia import *
from LecturaDatos import *
from LecturaDatosPacientes import *
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')

## ------------------------------------- CÁLCULO DE LAS ENERGÍAS --------------------------------------

## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Itero para cada uno de los identificadores de los pacientes. Hago la generación de escalogramas para cada paciente
for id_persona in ids_existentes:

    ## Coloco un bloque try en caso de que ocurra algún error de procesamiento
    try:

        ## ----------------------------------------- LECUTRA DE DATOS -----------------------------------------

        ## Hago la lectura de los datos del registro de marcha del paciente
        data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona)

        ## ------------------------------------- CÁLCULO DE LA ENERGÍA TOTAL -----------------------------------

        ## Obtengo la energía total de los registros de marcha del paciente
        energias_totales = EnergiaTotal(acel, gyro, id_persona)

        ## ------------------------------------- ESCRITURA EN BASE DE DATOS ------------------------------------

        ## Hago la lectura del archivo JSON de energías previamente existente
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias.json", 'r') as openfile:

            # Cargo el diccionario el cual va a ser un objeto JSON
            dicc_energias = json.load(openfile)

        ## Agrego en el diccionario los datos de energías del paciente actual
        dicc_energias[str(id_persona)] = energias_totales[id_persona]

        ## Especifico la ruta del archivo JSON sobre la cual voy a reescribir el diccionario de energías
        with open("C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Wavelets/energias.json", "w") as outfile:

            ## Escribo el diccionario actualizado
            json.dump(dicc_energias, outfile)
        
        ## Impresión en pantalla avisando el identificador del paciente que está siendo procesado
        print("ID del paciente que está siendo procesado: {}".format(id_persona))
    
    ## Si hay un error de procesamiento
    except:

        ## Que siga a la siguiente muestra
        continue