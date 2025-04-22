## ------------------------------------- IMPORTACIÓN DE LIBRERÍAS --------------------------------------

import pandas as pd
import numpy as np
import sys
import pandas as pd
from scipy import signal
from harm_analysis import *
from control import *
from LecturaDatos import *
from LecturaDatosPacientes import *

sys.path.append('C:/Yo/Tesis/SL2205-0.8/SL2205-0.8/gaitpy/gaitpy')
from gait import *

## ---------------------------------------- LECTURA DE DATOS -------------------------------------------

## Especifico el identificador del paciente para el cual voy a realizar la lectura de los datos
id_persona = 270

## Obtengo los datos leídos en el directorio correspondiente
data, acel, gyro, cant_muestras, periodoMuestreo, nombre_persona, nacimiento_persona = LecturaDatos(id_persona)

## --------------------------------------- DATOS DEL PACIENTE ------------------------------------------

## De todo el dataframe de pacientes me quedo únicamente con aquel paciente con el que estoy trabajando actualmente
## Debido al filtrado que se realiza previamente, ésto no producirá error ya que el paciente se encuentra en la base de datos
datos_paciente = pacientes[pacientes['sampleid'] == id_persona]

## -------------------------------------- PARÁMETROS DE MARCHA -----------------------------------------

## En esta parte se calculan los parámetros de marcha mediante el uso del algoritmo implementado previamente en GaitPy
## Expreso el vector de tiempos tomando como t = 0 el instante inicial de la sesión
## Se recuerda que los tiempos en dicho vector se encuentran expresados en milisegundos
tiempos = np.array(data['Time']) - data['Time'][0]

## Obtengo la aceleración vertical a efectos de ejecutar el algoritmo
acc_VT = np.array(data['AC_y'])

## Creo el dataframe cuyas columnas van a ser los instantes de tiempos y los valores de aceleración vertical respectivamente
## Por defecto el algoritmo toma el tiempo en milisegundos y las aceleraciones en metros por segundo cuadrado
dataframe_vert = pd.DataFrame(data = {'timestamps' : tiempos, 'y' : acc_VT})

## Creo un nuevo objeto de la clase GaitPy
gait_py = Gaitpy(data = dataframe_vert, sample_rate = 1 / periodoMuestreo)

## Obtengo las características de la marcha del sujeto. Debo pasar como parámetro la altura del sujeto en centímetros
features = gait_py.extract_features(subject_height = 190)

## Obtengo la lista de características que son extraídas de la señal de marcha
lista_features = features.columns

# lista_features = ['bout_number', 'bout_length_sec', 'bout_start_time', 'IC', 'FC',
#           'gait_cycles', 'steps', 'stride_duration', 'stride_duration_asymmetry',
#           'step_duration', 'step_duration_asymmetry', 'cadence',
#           'initial_double_support', 'initial_double_support_asymmetry',
#           'terminal_double_support', 'terminal_double_support_asymmetry',
#           'double_support', 'double_support_asymmetry', 'single_limb_support',
#           'single_limb_support_asymmetry', 'stance', 'stance_asymmetry', 'swing',
#           'swing_asymmetry', 'step_length', 'step_length_asymmetry',
#           'stride_length', 'stride_length_asymmetry', 'gait_speed']