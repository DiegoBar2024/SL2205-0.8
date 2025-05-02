## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import sys
sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/sereTestLib/Cinematica')
from LecturaDatos import *
from ParametrosGaitPy import *
from GeneracionReporte import *

## ------------------------------------- GENERACIÓN DEL REPORTE --------------------------------------

## Especifico el identificador del paciente para el cual quiero generar el reporte
id_persona = 270

## Hago la lectura de los datos generales de los pacientes
pacientes, ids_existentes = LecturaDatosPacientes()

## Hago la lectura de los datos del registro de marcha del paciente
data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona, tiempo = LecturaDatos(id_persona, lectura_datos_propios = True)

## Hago el cálculo de los parámetros de marcha
features, acc_VT = ParametrosMarcha(id_persona, data, periodoMuestreo, pacientes)

## Obtengo la longitud de los pasos asociada al registro de marcha del paciente
long_pasos = np.array(features['step_length'])

## Obtengo la duracion de los pasos asociada al registro de marcha del paciente
duraciones_pasos = np.array(features['step_duration'])

## Obtengo la velocidad de los pasos asociada al registro de marcha del paciente
velocidades = np.array(features['gait_speed'])

## Obtengo la cadencia de los pasos asociada al registro de marcha del paciente
frecuencias = np.array(features['cadence']) / 60

## Obtengo un vector que tenga los pasos numerados
pasos_numerados = np.arange(0, len(long_pasos), 1)

## Hago la creación del reporte en PDF asociado al paciente
CreacionReporte(id_persona, nombre_persona, nacimiento_persona, tiempo, long_pasos, duraciones_pasos, velocidades, frecuencias, pasos_numerados)